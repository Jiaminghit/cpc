from __future__ import annotations

"""对齐图像检测 pipeline。

输入是以 PCD timestamp 为主索引的 `aligned_index.json`，本流程会：
1. 找到每个 PCD-camera 对应的 RGB 图像；
2. 对唯一 RGB 图像去重推理，避免重复调用模型；
3. 将图像级检测结果展开回 PCD-camera 记录；
4. 写出 manifest、detections_by_image、detections_aligned 和 summary。
"""

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import torch

from models import build_model_adapter
from models.classes import CLASS_TABLE
from utils.alignment import iter_alignment_records, load_projection_lookup
from utils.detections import clone_detection_for_alignment
from utils.io import read_json, resolve_project_path, write_json


def run_aligned_detection(args: argparse.Namespace) -> None:
    """执行完整的 aligned model detection 流程。"""
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    aligned_index_path = Path(args.aligned_index).expanduser().resolve()
    projection_summary_path = (
        Path(args.projection_summary).expanduser().resolve() if args.projection_summary else None
    )
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = dataset_root / f"aligned_{args.model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    # aligned_index 提供 PCD timestamp 到各 camera RGB 图像的时间对齐关系。
    aligned_index = read_json(aligned_index_path)
    projection_lookup = load_projection_lookup(dataset_root, projection_summary_path)
    cameras = args.cameras or aligned_index.get("cameras", [])

    # 先展开出 PCD-camera 记录，后续再基于 rgb_image 做去重推理。
    aligned_records = list(
        iter_alignment_records(
            dataset_root=dataset_root,
            aligned_index=aligned_index,
            projection_lookup=projection_lookup,
            cameras=cameras,
            include_invalid=args.include_invalid,
            max_frames=args.max_frames,
        )
    )
    if not aligned_records:
        raise RuntimeError("No aligned image records selected.")

    # 同一张 RGB 可能对应多个邻近 PCD timestamp，只检测一次即可。
    unique_images = {}
    for record in aligned_records:
        unique_images.setdefault(record["rgb_image"], Path(record["rgb_image"]))
    if args.max_images is not None:
        allowed = set(list(unique_images.keys())[: args.max_images])
        aligned_records = [record for record in aligned_records if record["rgb_image"] in allowed]
        unique_images = {key: value for key, value in unique_images.items() if key in allowed}

    # adapter 屏蔽模型差异；pipeline 后续只依赖统一 predict/render 接口。
    adapter = build_model_adapter(args, resolve_project_path=resolve_project_path)

    manifest = {
        "schema_version": "aligned_model_detection_manifest.v1",
        "dataset_root": str(dataset_root),
        "aligned_index": str(aligned_index_path),
        "projection_summary": str(projection_summary_path) if projection_summary_path else None,
        "model": adapter.model_block(),
        "params": adapter.params_block(),
        "classes": CLASS_TABLE,
        "include_invalid": bool(args.include_invalid),
        "valid_frame_only": not bool(args.include_invalid),
        "cameras": cameras,
        "max_frames": args.max_frames,
        "max_images": args.max_images,
    }
    write_json(output_dir / "manifest.json", manifest)

    # 图像级输出：每个唯一 RGB 图像一行，保留模型原始检测视角。
    by_image_path = output_dir / "detections_by_image.jsonl"
    image_results: dict[str, dict[str, Any]] = {}
    with by_image_path.open("w", encoding="utf-8") as f:
        for idx, (image_key, image_path) in enumerate(unique_images.items(), start=1):
            result = adapter.predict(image_path=image_path, image_key=Path(image_key).stem)
            image_record = {
                "image_path": image_key,
                "image_id": image_path.stem,
                "camera": image_path.parent.name,
                "image_width": result["image_width"],
                "image_height": result["image_height"],
                "model": adapter.model_block(),
                "params": adapter.params_block(),
                "detections": result["detections"],
            }
            image_results[image_key] = image_record
            f.write(json.dumps(image_record, ensure_ascii=False) + "\n")
            print(f"[image {idx}/{len(unique_images)}] {image_key} detections={len(result['detections'])}")

            if args.save_vis:
                # 保存按 RGB timestamp 命名的可视化，便于快速按原图检查模型输出。
                image_vis_dir = output_dir / "vis_by_image" / image_path.parent.name
                image_vis_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_vis_dir / f"{image_path.stem}.jpg"), result["vis"])

    # 对齐级输出：把图像级检测复制到每个 PCD-camera 记录上，供后续 ROI/评估使用。
    aligned_path = output_dir / "detections_aligned.jsonl"
    total_detections = 0
    with aligned_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(aligned_records, start=1):
            image_record = image_results[record["rgb_image"]]
            detections = [
                clone_detection_for_alignment(
                    detection=det,
                    pcd_timestamp=record["pcd_timestamp"],
                    camera=record["camera"],
                    idx=det_idx,
                    model_name=adapter.model_name,
                )
                for det_idx, det in enumerate(image_record["detections"])
            ]
            aligned_record = {
                "dataset_id": record["dataset_id"],
                "pcd_timestamp": record["pcd_timestamp"],
                "camera": record["camera"],
                "rgb_timestamp": record["rgb_timestamp"],
                "image_delta_ms": record["image_delta_ms"],
                "valid_time_match": record["valid_time_match"],
                "alignment_reason": record["alignment_reason"],
                "frame_valid_time_match": record["frame_valid_time_match"],
                "frame_reasons": record["frame_reasons"],
                "rgb_image": record["rgb_image"],
                "rgb_image_rel": record["rgb_image_rel"],
                "projection_image": record["projection_image"],
                "projection": record["projection"],
                "image_width": image_record["image_width"],
                "image_height": image_record["image_height"],
                "model": adapter.model_block(),
                "params": adapter.params_block(),
                "detections": detections,
            }
            f.write(json.dumps(aligned_record, ensure_ascii=False) + "\n")
            total_detections += len(detections)

            if args.save_vis:
                image_path = Path(record["rgb_image"])
                annotated = adapter.render_detections(image_path, detections)
                vis_dir = output_dir / "vis" / record["camera"]
                vis_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(vis_dir / f"{record['pcd_timestamp']}.jpg"), annotated)

            print(
                f"[aligned {idx}/{len(aligned_records)}] "
                f"{record['pcd_timestamp']} {record['camera']} detections={len(detections)}"
            )

    summary = {
        "schema_version": "aligned_model_detection_summary.v1",
        "num_aligned_records": len(aligned_records),
        "num_unique_images": len(unique_images),
        "num_detections_expanded": total_detections,
        "output": {
            "manifest": str(output_dir / "manifest.json"),
            "detections_by_image": str(by_image_path),
            "detections_aligned": str(aligned_path),
        },
        "model": adapter.model_block(),
        "params": adapter.params_block(),
        "classes": CLASS_TABLE,
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Done. Wrote {aligned_path}")
