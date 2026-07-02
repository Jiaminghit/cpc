from __future__ import annotations

"""aligned_index 与投影摘要的读取/展开工具。"""

from pathlib import Path
from typing import Any, Iterable

from utils.io import read_json, resolve_dataset_path


def load_projection_lookup(
    dataset_root: Path,
    projection_summary_path: Path | None,
) -> dict[tuple[int, str], dict[str, Any]]:
    """读取 projection_summary，并按 `(pcd_timestamp, camera)` 建立快速索引。"""
    if not projection_summary_path:
        return {}
    data = read_json(projection_summary_path)
    lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for frame in data.get("frames", []):
        pcd_timestamp = int(frame["pcd_timestamp"])
        for camera, info in frame.get("cameras", {}).items():
            output_image = resolve_dataset_path(dataset_root, info.get("output_image"))
            lookup[(pcd_timestamp, camera)] = {
                "projection_image": str(output_image) if output_image else None,
                "projection": {
                    "boxes_total": info.get("boxes_total"),
                    "boxes_projected": info.get("boxes_projected"),
                    "boxes_skipped_behind_camera": info.get("boxes_skipped_behind_camera"),
                    "boxes_skipped_outside_image": info.get("boxes_skipped_outside_image"),
                    "skipped": info.get("skipped"),
                    "reason": info.get("reason"),
                },
            }
    return lookup


def iter_alignment_records(
    dataset_root: Path,
    aligned_index: dict[str, Any],
    projection_lookup: dict[tuple[int, str], dict[str, Any]],
    cameras: list[str],
    valid_time_only: bool,
    max_frames: int | None,
) -> Iterable[dict[str, Any]]:
    """把 aligned_index 展开为逐 PCD-camera 的检测输入记录。"""
    frames = aligned_index.get("frames", [])
    if max_frames is not None:
        frames = frames[:max_frames]

    dataset_id = dataset_root.name
    for frame in frames:
        pcd_timestamp = int(frame["pcd_timestamp"])
        frame_valid = bool(frame.get("valid_time_match"))
        for camera in cameras:
            image_info = frame.get("images", {}).get(camera)
            if not image_info:
                continue
            if not image_info.get("exists", True):
                continue
            valid_time_match = bool(image_info.get("valid_time_match"))
            if valid_time_only and not valid_time_match:
                continue

            # aligned_index 中的 image path 可能是绝对路径、数据集相对路径或带数据集名前缀的路径。
            image_path = resolve_dataset_path(dataset_root, image_info.get("path"))
            if image_path is None:
                continue

            projection_info = projection_lookup.get((pcd_timestamp, camera), {})
            yield {
                "dataset_id": dataset_id,
                "pcd_timestamp": pcd_timestamp,
                "camera": camera,
                "rgb_timestamp": image_info.get("timestamp"),
                "image_delta_ms": image_info.get("delta_ms"),
                "valid_time_match": valid_time_match,
                "alignment_reason": image_info.get("reason"),
                "frame_valid_time_match": frame_valid,
                "frame_reasons": frame.get("reasons", []),
                "rgb_image": str(image_path),
                "rgb_image_rel": image_info.get("path"),
                "projection_image": projection_info.get("projection_image"),
                "projection": projection_info.get("projection"),
                "calib": frame.get("calib"),
                "det_json": frame.get("det_json"),
            }
