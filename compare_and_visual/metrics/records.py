"""把一帧 projection/DINO 输入记录转换为一条评估结果记录。

utils.records 只负责通用 record 对齐和路径选择；本模块进入 metrics 业务层，
负责构造可匹配的 bbox、压缩输出字段、调用 matching，并生成 counts。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from metrics.aggregation import make_empty_counts
from metrics.bbox import normalize_bbox_xyxy, round_float
from metrics.matching import greedy_match
from utils.records import aligned_key, pick_rgb_image


def compact_roi(roi: dict[str, Any] | None) -> dict[str, Any]:
    """只保留 ROI 排查常用字段，避免 matches JSONL 过大。"""
    roi = roi or {}
    return {
        "in_roi": roi.get("in_roi"),
        "longitudinal_m": round_float(roi.get("longitudinal_m")),
        "lateral_m": round_float(roi.get("lateral_m")),
    }


def compact_projection_box(box: dict[str, Any], normalized_bbox: list[float] | None = None) -> dict[str, Any]:
    """压缩 projection box 输出，保留 ID、类别、2D/3D bbox 和 ROI 摘要。"""
    return {
        "projection_id": box.get("projection_id"),
        "object_id": box.get("object_id"),
        "class_name": box.get("class_name"),
        "class_id": box.get("class_id"),
        "score": round_float(box.get("score")),
        "bbox_xyxy": box.get("bbox_xyxy"),
        "bbox_xyxy_normalized": [round_float(v) for v in normalized_bbox] if normalized_bbox else None,
        "bbox_3d": {
            "center_ego": (box.get("bbox_3d") or {}).get("center_ego"),
            "size_lwh": (box.get("bbox_3d") or {}).get("size_lwh"),
            "heading": round_float((box.get("bbox_3d") or {}).get("heading")),
        },
        "roi": compact_roi(box.get("roi")),
    }


def compact_dino_box(box: dict[str, Any], normalized_bbox: list[float] | None = None) -> dict[str, Any]:
    """压缩 DINO detection 输出，字段结构尽量与 projection compact 对齐。"""
    return {
        "det_id": box.get("det_id"),
        "class_name": box.get("class_name"),
        "class_id": box.get("class_id"),
        "score": round_float(box.get("score")),
        "bbox_xyxy": box.get("bbox_xyxy"),
        "bbox_xyxy_normalized": [round_float(v) for v in normalized_bbox] if normalized_bbox else None,
        "lidar_roi": compact_roi(box.get("lidar_roi")),
    }


def record_image_size(projection: dict[str, Any] | None, dino: dict[str, Any] | None) -> tuple[int | None, int | None]:
    """从任一侧记录中读取图像尺寸，用于 bbox 裁剪和有效性判断。"""
    for record in (projection, dino):
        if not record:
            continue
        width = record.get("image_width")
        height = record.get("image_height")
        if isinstance(width, int) and isinstance(height, int):
            return width, height
    return None, None


def build_projection_boxes(
    record: dict[str, Any] | None,
    image_width: int | None,
    image_height: int | None,
    warnings: list[str],
    require_roi: bool,
) -> list[dict[str, Any]]:
    """从 projection record 提取可参与匹配的 box 列表。"""
    if record is None:
        return []
    boxes = []
    for index, box in enumerate(record.get("projected_boxes", [])):
        roi = box.get("roi") or {}
        if require_roi and roi.get("in_roi") is not True:
            # ROI-filtered 输入理论上都应在 ROI 内；不直接丢弃，但记录 warning。
            warnings.append(
                f"projection_non_roi_box:{record.get('pcd_timestamp')}:{record.get('camera')}:{index}"
            )
        normalized_bbox = normalize_bbox_xyxy(box.get("bbox_xyxy"), image_width, image_height)
        if normalized_bbox is None:
            warnings.append(
                f"projection_invalid_bbox:{record.get('pcd_timestamp')}:{record.get('camera')}:{index}"
            )
            continue
        boxes.append(
            {
                "index": index,
                "raw": box,
                "bbox": normalized_bbox,
                "compact": compact_projection_box(box, normalized_bbox),
            }
        )
    return boxes


def build_dino_boxes(
    record: dict[str, Any] | None,
    image_width: int | None,
    image_height: int | None,
    warnings: list[str],
    require_roi: bool,
) -> list[dict[str, Any]]:
    """从 DINO record 提取可参与匹配的 detection 列表。"""
    if record is None:
        return []
    boxes = []
    for index, box in enumerate(record.get("detections", [])):
        roi = box.get("lidar_roi") or {}
        if require_roi and roi.get("in_roi") is not True:
            warnings.append(
                f"dino_non_roi_box:{record.get('pcd_timestamp')}:{record.get('camera')}:{index}"
            )
        normalized_bbox = normalize_bbox_xyxy(box.get("bbox_xyxy"), image_width, image_height)
        if normalized_bbox is None:
            warnings.append(f"dino_invalid_bbox:{record.get('pcd_timestamp')}:{record.get('camera')}:{index}")
            continue
        boxes.append(
            {
                "index": index,
                "raw": box,
                "bbox": normalized_bbox,
                "compact": compact_dino_box(box, normalized_bbox),
            }
        )
    return boxes


def base_record_fields(
    dataset_root: Path,
    projection: dict[str, Any] | None,
    dino: dict[str, Any] | None,
) -> dict[str, Any]:
    """整理输出 record 的公共元信息，优先使用可解析到本地文件的图片路径。"""
    base = projection or dino or {}
    image_path = pick_rgb_image(dataset_root, projection, dino)
    return {
        "dataset_id": base.get("dataset_id") or dataset_root.name,
        "pcd_timestamp": base.get("pcd_timestamp"),
        "camera": base.get("camera"),
        "rgb_timestamp": base.get("rgb_timestamp"),
        "rgb_image": str(image_path) if image_path else base.get("rgb_image"),
        "valid_time_match": base.get("valid_time_match"),
        "frame_valid_time_match": base.get("frame_valid_time_match"),
        "image_width": base.get("image_width"),
        "image_height": base.get("image_height"),
    }


def evaluate_record(
    *,
    dataset_root: Path,
    projection: dict[str, Any] | None,
    dino: dict[str, Any] | None,
    iou_threshold: float,
    low_iou_threshold: float,
    class_aware_matching: bool,
    require_roi: bool,
    warnings: list[str],
) -> dict[str, Any]:
    """评估单个 pcd_timestamp + camera 下的 projection/DINO 一致性。"""
    base = projection or dino
    if base is None:
        raise ValueError("projection and dino cannot both be None")
    pcd_timestamp, camera = aligned_key(base)
    image_width, image_height = record_image_size(projection, dino)
    # 两侧 box 会被归一化成统一的内部结构，matching 不需要关心原始字段差异。
    projection_boxes = build_projection_boxes(
        projection,
        image_width,
        image_height,
        warnings,
        require_roi=require_roi,
    )
    dino_boxes = build_dino_boxes(
        dino,
        image_width,
        image_height,
        warnings,
        require_roi=require_roi,
    )
    matches, unmatched_projection, unmatched_dino = greedy_match(
        projection_boxes,
        dino_boxes,
        iou_threshold=iou_threshold,
        low_iou_threshold=low_iou_threshold,
        class_aware_matching=class_aware_matching,
        pcd_timestamp=pcd_timestamp,
        camera=camera,
    )

    # counts 是 record 级 summary；完整匹配明细仍保存在 matches/unmatched 中。
    counts = make_empty_counts()
    counts["projection_boxes"] = len(projection_boxes)
    counts["dino_boxes"] = len(dino_boxes)
    counts["matches"] = len(matches)
    for match in matches:
        counts[match["match_status"]] += 1
    counts["unmatched_projection"] = len(unmatched_projection)
    counts["unmatched_dino"] = len(unmatched_dino)

    record = {
        "schema_version": "roi_projection_dino_matches.v1",
        **base_record_fields(dataset_root, projection, dino),
        "counts": counts,
        "matches": matches,
        "unmatched_projection_boxes": unmatched_projection,
        "unmatched_dino_boxes": unmatched_dino,
    }
    return record
