"""ROI 投影框 vs DINO 检测框的通用评估主流程。

pipeline 层负责串起 IO、记录对齐、逐帧匹配、聚合统计和输出文件写入；
具体 bbox 几何、matching、aggregation 细节分别下沉到 metrics 子模块。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from metrics.aggregation import (
    finalize_aggregate,
    make_aggregate,
    update_aggregate_from_record,
    update_class_aggregate,
)
from metrics.records import evaluate_record
from tools.error_visualization import import_cv2, save_error_visualization
from utils.classes import CLASS_TABLE
from utils.io import read_jsonl, write_json, write_jsonl_line
from utils.records import index_records, select_comparable_keys


def validate_args(args: argparse.Namespace) -> None:
    """检查阈值组合，避免后续指标含义变得不明确。"""
    if args.low_iou_threshold < 0.0:
        raise ValueError("--low-iou-threshold must be >= 0.")
    if args.iou_threshold <= 0.0 or args.iou_threshold > 1.0:
        raise ValueError("--iou-threshold must be in (0, 1].")
    if args.low_iou_threshold > args.iou_threshold:
        raise ValueError("--low-iou-threshold must be <= --iou-threshold.")


def build_manifest(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    projection_jsonl: Path,
    dino_jsonl: Path,
    output_dir: Path,
    matches_path: Path,
    unmatched_projection_path: Path,
    unmatched_dino_path: Path,
) -> dict[str, Any]:
    """生成一次评估运行的 manifest，方便追踪输入、参数和输出文件。"""
    return {
        "schema_version": "roi_projection_dino_eval_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "projection_jsonl": str(projection_jsonl),
        "dino_jsonl": str(dino_jsonl),
        "output_dir": str(output_dir),
        "classes": CLASS_TABLE,
        "params": {
            "iou_threshold": args.iou_threshold,
            "low_iou_threshold": args.low_iou_threshold,
            "matching_method": "greedy",
            "class_aware_matching": bool(args.class_aware_matching),
            "allow_class_mismatch": not bool(args.class_aware_matching),
            "include_invalid": bool(args.include_invalid),
            "valid_frame_only": not bool(args.include_invalid),
            "key_selection": "intersection",
            "input_scope": args.input_scope,
            "require_roi_fields": args.input_scope == "roi_filtered",
            "max_records": args.max_records,
            "cameras": args.cameras,
            "save_error_vis": bool(args.save_error_vis),
        },
        "outputs": {
            "matches_aligned": str(matches_path),
            "metrics_summary": str(output_dir / "metrics_summary.json"),
            "metrics_by_camera": str(output_dir / "metrics_by_camera.json"),
            "metrics_by_class": str(output_dir / "metrics_by_class.json"),
            "unmatched_projection_boxes": str(unmatched_projection_path),
            "unmatched_dino_boxes": str(unmatched_dino_path),
            "error_vis": str(output_dir / "error_vis") if args.save_error_vis else None,
        },
    }


def write_unmatched_records(
    *,
    record: dict[str, Any],
    unmatched_projection_writer: Any,
    unmatched_dino_writer: Any,
) -> None:
    """把 unmatched 明细拆成两个 JSONL，方便单独检查漏检/多检。"""
    for item in record["unmatched_projection_boxes"]:
        write_jsonl_line(
            unmatched_projection_writer,
            {
                "pcd_timestamp": record["pcd_timestamp"],
                "camera": record["camera"],
                "rgb_image": record.get("rgb_image"),
                **item,
            },
        )
    for item in record["unmatched_dino_boxes"]:
        write_jsonl_line(
            unmatched_dino_writer,
            {
                "pcd_timestamp": record["pcd_timestamp"],
                "camera": record["camera"],
                "rgb_image": record.get("rgb_image"),
                **item,
            },
        )


def run_projection_model_eval(args: argparse.Namespace) -> None:
    """执行完整评估：读取输入、逐帧匹配、写明细、汇总全局/相机/类别指标。"""
    validate_args(args)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    projection_jsonl = Path(args.projection_jsonl).expanduser().resolve()
    dino_jsonl = Path(args.dino_jsonl).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else dataset_root / "compare_grounding_dino_b_roi_projection_qc"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2 = import_cv2() if args.save_error_vis else None

    # 先把两路 JSONL 建成同一种 key 索引，后面只遍历正式可比较的 key。
    projection_records = read_jsonl(projection_jsonl)
    dino_records = read_jsonl(dino_jsonl)
    projection_by_key = index_records(projection_records)
    dino_by_key = index_records(dino_records)
    keys, key_stats = select_comparable_keys(
        projection_by_key=projection_by_key,
        model_by_key=dino_by_key,
        cameras=args.cameras,
        include_invalid=bool(args.include_invalid),
        max_records=args.max_records,
    )

    # 三套聚合分别对应全局、按相机、按类别的 summary 输出。
    warnings: list[str] = []
    global_aggregate = make_aggregate()
    camera_aggregates: dict[str, dict[str, Any]] = defaultdict(make_aggregate)
    class_aggregates: dict[str, dict[str, Any]] = defaultdict(make_aggregate)
    missing_projection_records = 0
    missing_dino_records = 0
    error_vis_written = 0

    matches_path = output_dir / "matches_aligned.jsonl"
    unmatched_projection_path = output_dir / "unmatched_projection_boxes.jsonl"
    unmatched_dino_path = output_dir / "unmatched_dino_boxes.jsonl"

    manifest = build_manifest(
        args=args,
        dataset_root=dataset_root,
        projection_jsonl=projection_jsonl,
        dino_jsonl=dino_jsonl,
        output_dir=output_dir,
        matches_path=matches_path,
        unmatched_projection_path=unmatched_projection_path,
        unmatched_dino_path=unmatched_dino_path,
    )
    manifest["key_selection"] = key_stats
    write_json(output_dir / "manifest.json", manifest)

    with matches_path.open("w", encoding="utf-8") as matches_writer, unmatched_projection_path.open(
        "w", encoding="utf-8"
    ) as unmatched_projection_writer, unmatched_dino_path.open("w", encoding="utf-8") as unmatched_dino_writer:
        for index, key in enumerate(keys, start=1):
            projection = projection_by_key.get(key)
            dino = dino_by_key.get(key)
            if projection is None:
                missing_projection_records += 1
            if dino is None:
                missing_dino_records += 1

            # evaluate_record 只处理一帧/一相机，pipeline 负责循环和输出。
            record = evaluate_record(
                dataset_root=dataset_root,
                projection=projection,
                dino=dino,
                iou_threshold=args.iou_threshold,
                low_iou_threshold=args.low_iou_threshold,
                class_aware_matching=bool(args.class_aware_matching),
                require_roi=args.input_scope == "roi_filtered",
                warnings=warnings,
            )
            write_jsonl_line(matches_writer, record)
            write_unmatched_records(
                record=record,
                unmatched_projection_writer=unmatched_projection_writer,
                unmatched_dino_writer=unmatched_dino_writer,
            )

            update_aggregate_from_record(global_aggregate, record)
            update_aggregate_from_record(camera_aggregates[str(record["camera"])], record)
            update_class_aggregate(class_aggregates, record)

            # error vis 只为非 TP 状态落图；TP-only 记录不会产生可视化文件。
            if args.save_error_vis and cv2 is not None:
                written = save_error_visualization(
                    cv2=cv2,
                    dataset_root=dataset_root,
                    output_dir=output_dir,
                    record=record,
                    projection_record=projection,
                    dino_record=dino,
                )
                error_vis_written += len(written)

            print(
                f"[{index}/{len(keys)}] {record['pcd_timestamp']} {record['camera']} "
                f"proj={record['counts']['projection_boxes']} dino={record['counts']['dino_boxes']} "
                f"tp={record['counts']['tp']} cls_mis={record['counts']['class_mismatch']} "
                f"unmatched_ref={record['counts']['unmatched_projection']} "
                f"unmatched_dino={record['counts']['unmatched_dino']}"
            )

    # finalize 会把中间累积列表压成均值/中位数等可直接阅读的指标。
    summary_metrics = finalize_aggregate(global_aggregate)
    summary = {
        "schema_version": "roi_projection_dino_metrics_summary.v1",
        "dataset_root": str(dataset_root),
        "projection_jsonl": str(projection_jsonl),
        "dino_jsonl": str(dino_jsonl),
        "output_dir": str(output_dir),
        "params": manifest["params"],
        "records": {
            "projection_records": len(projection_records),
            "dino_records": len(dino_records),
            "aligned_records": len(keys),
            "key_selection": key_stats,
            "records_with_projection_only": summary_metrics["records_with_projection_only"],
            "records_with_dino_only": summary_metrics["records_with_dino_only"],
            "missing_projection_records": missing_projection_records,
            "missing_dino_records": missing_dino_records,
        },
        "boxes": summary_metrics["boxes"],
        "matches": summary_metrics["matches"],
        "metrics": summary_metrics["metrics"],
        "class_mismatch_pairs": summary_metrics["class_mismatch_pairs"],
        "error_vis_written": error_vis_written if args.save_error_vis else 0,
        "warnings": sorted(set(warnings + key_stats["warnings"])),
        "outputs": manifest["outputs"],
    }

    by_camera = {
        camera: finalize_aggregate(block)
        for camera, block in sorted(camera_aggregates.items())
    }
    by_class = {
        class_name: finalize_aggregate(block)
        for class_name, block in sorted(class_aggregates.items())
    }

    write_json(output_dir / "metrics_summary.json", summary)
    write_json(
        output_dir / "metrics_by_camera.json",
        {"schema_version": "roi_projection_dino_metrics_by_camera.v1", "by_camera": by_camera},
    )
    write_json(
        output_dir / "metrics_by_class.json",
        {"schema_version": "roi_projection_dino_metrics_by_class.v1", "by_class": by_class},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
