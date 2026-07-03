"""投影框和模型检测框的叠加可视化 pipeline。

本流程只做“看图检查”：按 pcd_timestamp + camera 对齐记录，生成 overlay
图片和统计摘要，不参与 IoU/matching 等指标计算。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tools.overlay import overlay_record
from utils.classes import CLASS_TABLE
from utils.io import read_jsonl, write_json
from utils.records import index_records, select_comparable_keys


def run_projection_model_overlay(args: argparse.Namespace) -> None:
    """执行完整 overlay 流程：读取记录、对齐、绘图、汇总每个相机的输出情况。"""
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    projection_jsonl = Path(args.projection_jsonl).expanduser().resolve()
    model_jsonl = Path(args.model_jsonl).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else dataset_root / "compare_grounding_dino_b_projection"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 与评估流程保持一致：先按同一 key 建索引，再遍历正式可比较的 key。
    projection_records = read_jsonl(projection_jsonl)
    model_records = read_jsonl(model_jsonl)
    projection_by_key = index_records(projection_records)
    model_by_key = index_records(model_records)
    keys, key_stats = select_comparable_keys(
        projection_by_key=projection_by_key,
        model_by_key=model_by_key,
        cameras=args.cameras,
        include_invalid=bool(args.include_invalid),
        max_records=args.max_records,
    )

    # manifest 用来记录本次可视化的输入、参数和输出位置，便于复现实验。
    manifest = {
        "schema_version": "overlay_manifest.v1",
        "dataset_root": str(dataset_root),
        "projection_jsonl": str(projection_jsonl),
        "model_jsonl": str(model_jsonl),
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "classes": CLASS_TABLE,
        "save_vis": args.save_vis,
        "draw_projection_corners": args.draw_projection_corners,
        "model_mask_alpha": args.model_mask_alpha,
        "include_invalid": bool(args.include_invalid),
        "valid_frame_only": not bool(args.include_invalid),
        "key_selection": key_stats,
        "max_records": args.max_records,
        "cameras": args.cameras,
        "note": "Stage 3A visualization only; no IoU or TP/FP/FN is computed.",
    }
    write_json(output_dir / "manifest.json", manifest)

    records_path = output_dir / "overlay_records.jsonl"
    camera_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "records": 0,
            "written": 0,
            "skipped": 0,
            "reference_boxes": 0,
            "model_boxes": 0,
        }
    )
    total_reference_boxes = 0
    total_model_boxes = 0
    total_written = 0
    total_skipped = 0
    missing_projection_records = 0
    missing_model_records = 0

    with records_path.open("w", encoding="utf-8") as writer:
        for idx, key in enumerate(keys, start=1):
            projection = projection_by_key.get(key)
            model = model_by_key.get(key)
            if projection is None:
                missing_projection_records += 1
            if model is None:
                missing_model_records += 1
            pcd_timestamp, camera = key
            output_path = output_dir / "vis" / camera / f"{pcd_timestamp}.jpg"
            # overlay_record 只处理单帧绘图，pipeline 负责循环、计数和文件组织。
            overlay = overlay_record(
                dataset_root=dataset_root,
                projection=projection,
                model=model,
                output_path=output_path,
                save_vis=args.save_vis,
                draw_projection_corners=args.draw_projection_corners,
                model_mask_alpha=args.model_mask_alpha,
            )
            writer.write(json.dumps(overlay, ensure_ascii=False) + "\n")

            camera_counts[camera]["records"] += 1
            camera_counts[camera]["reference_boxes"] += overlay["reference_box_count"]
            camera_counts[camera]["model_boxes"] += overlay["model_box_count"]
            total_reference_boxes += overlay["reference_box_count"]
            total_model_boxes += overlay["model_box_count"]
            if overlay["status"] == "ok":
                camera_counts[camera]["written"] += 1
                total_written += 1
            else:
                camera_counts[camera]["skipped"] += 1
                total_skipped += 1

            print(
                f"[{idx}/{len(keys)}] {pcd_timestamp} {camera} "
                f"ref={overlay['reference_box_count']} model={overlay['model_box_count']} "
                f"status={overlay['status']}"
            )

    # summary 只统计可视化层面的成功/跳过数量，不表达模型质量指标。
    summary = {
        "schema_version": "overlay_summary.v1",
        "stage": "3A_visual_overlay_only",
        "num_projection_records": len(projection_records),
        "num_model_records": len(model_records),
        "num_selected_records": len(keys),
        "key_selection": key_stats,
        "num_overlay_images_written": total_written if args.save_vis else 0,
        "num_skipped_records": total_skipped,
        "num_missing_projection_records": missing_projection_records,
        "num_missing_model_records": missing_model_records,
        "total_reference_boxes": total_reference_boxes,
        "total_model_boxes": total_model_boxes,
        "per_camera": dict(sorted(camera_counts.items())),
        "output": {
            "manifest": str(output_dir / "manifest.json"),
            "overlay_summary": str(output_dir / "overlay_summary.json"),
            "overlay_records": str(records_path),
            "vis": str(output_dir / "vis"),
        },
        "warnings": sorted(set(key_stats["warnings"])),
        "note": "No IoU, matching, TP/FP/FN, precision, or recall is computed in this stage.",
    }
    write_json(output_dir / "overlay_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
