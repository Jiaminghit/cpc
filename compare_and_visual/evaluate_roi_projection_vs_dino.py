#!/usr/bin/env python3
"""Evaluate ROI 2D consistency between projected 3D boxes and DINO boxes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MATCH_KEYS = [
    "tp",
    "class_mismatch",
    "low_iou_matched",
    "weak_overlap_matched",
    "unmatched_projection",
    "unmatched_dino",
]

CLASS_TABLE = [
    {"class_id": 0, "class_name": "Car"},
    {"class_id": 1, "class_name": "Pedestrian"},
    {"class_id": 2, "class_name": "Cyclist"},
    {"class_id": 3, "class_name": "Van"},
    {"class_id": 4, "class_name": "Traffic_cone"},
]

VIS_COLORS = {
    "tp": (60, 220, 60),
    "class_mismatch": (255, 80, 255),
    "low_iou_matched": (40, 220, 255),
    "weak_overlap_matched": (40, 180, 255),
    "unmatched_projection": (40, 40, 255),
    "unmatched_dino": (255, 120, 40),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ROI 2D consistency between projected 3D boxes and GroundingDINO ROI detections."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root directory.")
    parser.add_argument("--projection-jsonl", required=True, help="ROI projections_aligned.jsonl path.")
    parser.add_argument("--dino-jsonl", required=True, help="ROI detections_aligned_roi_only.jsonl path.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <dataset-root>/compare_grounding_dino_b_roi_projection_qc.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--low-iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument(
        "--class-aware-matching",
        action="store_true",
        help="Only allow candidate pairs with the same class_name.",
    )
    parser.add_argument(
        "--allow-class-mismatch",
        action="store_true",
        default=True,
        help="Compatibility flag; class mismatch is allowed by default unless --class-aware-matching is set.",
    )
    parser.add_argument(
        "--valid-time-only",
        action="store_true",
        help="Evaluate only records whose valid_time_match or frame_valid_time_match is true.",
    )
    parser.add_argument(
        "--save-error-vis",
        action="store_true",
        help="Save lightweight visualizations for non-TP records.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as reader:
        for line_no, line in enumerate(reader, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_line_no"] = line_no
            records.append(record)
    return records


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl_line(writer: Any, data: dict[str, Any]) -> None:
    writer.write(json.dumps(data, ensure_ascii=False) + "\n")


def aligned_key(record: dict[str, Any]) -> tuple[int, str]:
    return int(record["pcd_timestamp"]), str(record["camera"])


def index_records(records: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for record in records:
        indexed[aligned_key(record)] = record
    return indexed


def resolve_path(dataset_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == dataset_root.name:
        return dataset_root.parent / path
    return dataset_root / path


def pick_rgb_image(
    dataset_root: Path,
    projection: dict[str, Any] | None,
    dino: dict[str, Any] | None,
) -> Path | None:
    for record in (projection, dino):
        if not record:
            continue
        image = resolve_path(dataset_root, record.get("rgb_image"))
        if image and image.is_file():
            return image
        image_rel = resolve_path(dataset_root, record.get("rgb_image_rel"))
        if image_rel and image_rel.is_file():
            return image_rel
    return None


def is_valid_time_record(record: dict[str, Any] | None) -> bool:
    if record is None:
        return False
    return bool(record.get("valid_time_match") or record.get("frame_valid_time_match"))


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def round_float(value: Any, digits: int = 6) -> float | None:
    number = finite_float(value)
    if number is None:
        return None
    return round(number, digits)


def normalize_bbox_xyxy(
    bbox: Any,
    image_width: int | None,
    image_height: int | None,
) -> list[float] | None:
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return None
    values = [finite_float(value) for value in bbox]
    if any(value is None for value in values):
        return None
    x1, y1, x2, y2 = [float(value) for value in values]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if image_width is not None and image_width > 0:
        x1 = min(max(x1, 0.0), float(image_width - 1))
        x2 = min(max(x2, 0.0), float(image_width - 1))
    if image_height is not None and image_height > 0:
        y1 = min(max(y1, 0.0), float(image_height - 1))
        y2 = min(max(y2, 0.0), float(image_height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = bbox_area(box_a)
    area_b = bbox_area(box_b)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def center_distance_px(box_a: list[float], box_b: list[float]) -> float:
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    return math.hypot(ax - bx, ay - by)


def area_ratio_projection_over_dino(projection_box: list[float], dino_box: list[float]) -> float | None:
    dino_area = bbox_area(dino_box)
    if dino_area <= 0.0:
        return None
    return bbox_area(projection_box) / dino_area


def metric_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.fmean(values)), 6)


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.median(values)), 6)


def make_empty_counts() -> dict[str, int]:
    counts = {
        "projection_boxes": 0,
        "dino_boxes": 0,
        "matches": 0,
    }
    for key in MATCH_KEYS:
        counts[key] = 0
    return counts


def make_aggregate() -> dict[str, Any]:
    return {
        "records": 0,
        "records_with_projection_only": 0,
        "records_with_dino_only": 0,
        "boxes": {"projection_boxes": 0, "dino_boxes": 0},
        "matches": {key: 0 for key in MATCH_KEYS},
        "iou_values": [],
        "center_distance_px_values": [],
        "class_mismatch_pairs": defaultdict(int),
    }


def finalize_aggregate(block: dict[str, Any]) -> dict[str, Any]:
    matches = dict(block["matches"])
    geometry_tp = matches["tp"] + matches["class_mismatch"]
    precision = metric_ratio(geometry_tp, geometry_tp + matches["unmatched_projection"])
    recall = metric_ratio(geometry_tp, geometry_tp + matches["unmatched_dino"])
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = round(2.0 * precision * recall / (precision + recall), 6)
    return {
        "records": block["records"],
        "records_with_projection_only": block["records_with_projection_only"],
        "records_with_dino_only": block["records_with_dino_only"],
        "boxes": dict(block["boxes"]),
        "matches": matches,
        "metrics": {
            "pseudo_precision": precision,
            "pseudo_recall": recall,
            "pseudo_f1": f1,
            "mean_iou_matched": mean_or_none(block["iou_values"]),
            "median_iou_matched": median_or_none(block["iou_values"]),
            "mean_center_distance_px": mean_or_none(block["center_distance_px_values"]),
        },
        "class_mismatch_pairs": dict(sorted(block["class_mismatch_pairs"].items())),
    }


def compact_roi(roi: dict[str, Any] | None) -> dict[str, Any]:
    roi = roi or {}
    return {
        "in_roi": roi.get("in_roi"),
        "longitudinal_m": round_float(roi.get("longitudinal_m")),
        "lateral_m": round_float(roi.get("lateral_m")),
    }


def compact_projection_box(box: dict[str, Any], normalized_bbox: list[float] | None = None) -> dict[str, Any]:
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
) -> list[dict[str, Any]]:
    if record is None:
        return []
    boxes = []
    for index, box in enumerate(record.get("projected_boxes", [])):
        roi = box.get("roi") or {}
        if roi.get("in_roi") is not True:
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
) -> list[dict[str, Any]]:
    if record is None:
        return []
    boxes = []
    for index, box in enumerate(record.get("detections", [])):
        roi = box.get("lidar_roi") or {}
        if roi.get("in_roi") is not True:
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


def match_status_for(iou: float, class_equal: bool, iou_threshold: float, low_iou_threshold: float) -> str:
    if iou >= iou_threshold and class_equal:
        return "tp"
    if iou >= iou_threshold and not class_equal:
        return "class_mismatch"
    if iou >= low_iou_threshold:
        return "low_iou_matched"
    return "weak_overlap_matched"


def best_iou_for_unmatched(
    source_bbox: list[float],
    other_boxes: list[dict[str, Any]],
    other_id_key: str,
) -> tuple[float | None, str | None]:
    best_iou = 0.0
    best_id = None
    for other in other_boxes:
        iou = bbox_iou(source_bbox, other["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_id = other["raw"].get(other_id_key)
    return (round(best_iou, 6), best_id) if best_id is not None else (None, None)


def greedy_match(
    projection_boxes: list[dict[str, Any]],
    dino_boxes: list[dict[str, Any]],
    *,
    iou_threshold: float,
    low_iou_threshold: float,
    class_aware_matching: bool,
    pcd_timestamp: int,
    camera: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = []
    for projection in projection_boxes:
        for dino in dino_boxes:
            class_equal = projection["raw"].get("class_name") == dino["raw"].get("class_name")
            if class_aware_matching and not class_equal:
                continue
            iou = bbox_iou(projection["bbox"], dino["bbox"])
            if iou <= 0.0:
                continue
            distance = center_distance_px(projection["bbox"], dino["bbox"])
            candidates.append(
                {
                    "projection_index": projection["index"],
                    "dino_index": dino["index"],
                    "projection": projection,
                    "dino": dino,
                    "iou": iou,
                    "center_distance_px": distance,
                    "area_ratio_projection_over_dino": area_ratio_projection_over_dino(
                        projection["bbox"], dino["bbox"]
                    ),
                    "class_equal": class_equal,
                }
            )
    candidates.sort(key=lambda item: (-item["iou"], item["center_distance_px"]))

    used_projection: set[int] = set()
    used_dino: set[int] = set()
    matches = []
    for candidate in candidates:
        projection_index = candidate["projection_index"]
        dino_index = candidate["dino_index"]
        if projection_index in used_projection or dino_index in used_dino:
            continue
        used_projection.add(projection_index)
        used_dino.add(dino_index)
        status = match_status_for(
            candidate["iou"],
            candidate["class_equal"],
            iou_threshold,
            low_iou_threshold,
        )
        projection = candidate["projection"]
        dino = candidate["dino"]
        match_id = f"match:{pcd_timestamp}:{camera}:{len(matches)}"
        matches.append(
            {
                "match_id": match_id,
                "match_status": status,
                "iou": round(candidate["iou"], 6),
                "center_distance_px": round(candidate["center_distance_px"], 6),
                "area_ratio_projection_over_dino": round_float(
                    candidate["area_ratio_projection_over_dino"]
                ),
                "class_equal": candidate["class_equal"],
                "projection": projection["compact"],
                "dino": dino["compact"],
            }
        )

    unmatched_projection = []
    for projection in projection_boxes:
        if projection["index"] in used_projection:
            continue
        best_iou, best_det_id = best_iou_for_unmatched(projection["bbox"], dino_boxes, "det_id")
        unmatched_projection.append(
            {
                "match_status": "unmatched_projection",
                "best_iou": best_iou,
                "best_dino_det_id": best_det_id,
                "projection": projection["compact"],
            }
        )

    unmatched_dino = []
    for dino in dino_boxes:
        if dino["index"] in used_dino:
            continue
        best_iou, best_projection_id = best_iou_for_unmatched(dino["bbox"], projection_boxes, "projection_id")
        unmatched_dino.append(
            {
                "match_status": "unmatched_dino",
                "best_iou": best_iou,
                "best_projection_id": best_projection_id,
                "dino": dino["compact"],
            }
        )

    return matches, unmatched_projection, unmatched_dino


def base_record_fields(
    dataset_root: Path,
    projection: dict[str, Any] | None,
    dino: dict[str, Any] | None,
) -> dict[str, Any]:
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
    warnings: list[str],
) -> dict[str, Any]:
    base = projection or dino
    if base is None:
        raise ValueError("projection and dino cannot both be None")
    pcd_timestamp, camera = aligned_key(base)
    image_width, image_height = record_image_size(projection, dino)
    projection_boxes = build_projection_boxes(projection, image_width, image_height, warnings)
    dino_boxes = build_dino_boxes(dino, image_width, image_height, warnings)
    matches, unmatched_projection, unmatched_dino = greedy_match(
        projection_boxes,
        dino_boxes,
        iou_threshold=iou_threshold,
        low_iou_threshold=low_iou_threshold,
        class_aware_matching=class_aware_matching,
        pcd_timestamp=pcd_timestamp,
        camera=camera,
    )

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


def update_aggregate_from_record(block: dict[str, Any], record: dict[str, Any]) -> None:
    counts = record["counts"]
    block["records"] += 1
    if counts["projection_boxes"] > 0 and counts["dino_boxes"] == 0:
        block["records_with_projection_only"] += 1
    if counts["dino_boxes"] > 0 and counts["projection_boxes"] == 0:
        block["records_with_dino_only"] += 1
    block["boxes"]["projection_boxes"] += counts["projection_boxes"]
    block["boxes"]["dino_boxes"] += counts["dino_boxes"]
    for key in MATCH_KEYS:
        block["matches"][key] += counts[key]
    for match in record["matches"]:
        block["iou_values"].append(float(match["iou"]))
        block["center_distance_px_values"].append(float(match["center_distance_px"]))
        if match["match_status"] == "class_mismatch":
            projection_class = match["projection"].get("class_name") or "unknown"
            dino_class = match["dino"].get("class_name") or "unknown"
            block["class_mismatch_pairs"][f"{projection_class}->{dino_class}"] += 1


def update_class_aggregate(
    blocks: dict[str, dict[str, Any]],
    record: dict[str, Any],
) -> None:
    touched_classes: set[str] = set()
    for match in record["matches"]:
        projection_class = match["projection"].get("class_name") or "unknown"
        touched_classes.add(projection_class)
        block = blocks[projection_class]
        block["boxes"]["projection_boxes"] += 1
        block["boxes"]["dino_boxes"] += 1
        block["matches"][match["match_status"]] += 1
        block["iou_values"].append(float(match["iou"]))
        block["center_distance_px_values"].append(float(match["center_distance_px"]))
        if match["match_status"] == "class_mismatch":
            dino_class = match["dino"].get("class_name") or "unknown"
            block["class_mismatch_pairs"][f"{projection_class}->{dino_class}"] += 1

    for item in record["unmatched_projection_boxes"]:
        class_name = item["projection"].get("class_name") or "unknown"
        touched_classes.add(class_name)
        block = blocks[class_name]
        block["boxes"]["projection_boxes"] += 1
        block["matches"]["unmatched_projection"] += 1

    for item in record["unmatched_dino_boxes"]:
        class_name = item["dino"].get("class_name") or "unknown"
        touched_classes.add(class_name)
        block = blocks[class_name]
        block["boxes"]["dino_boxes"] += 1
        block["matches"]["unmatched_dino"] += 1

    for class_name in touched_classes:
        blocks[class_name]["records"] += 1


def import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("--save-error-vis requires cv2/opencv-python in the active environment") from exc
    return cv2


def draw_label(cv2: Any, image: Any, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = int(min(max(origin[0], 0), max(0, width - text_w - 2)))
    y = int(min(max(origin[1], text_h + baseline + 2), height - 2))
    cv2.rectangle(
        image,
        (x, y - text_h - baseline - 3),
        (x + text_w + 4, y + baseline + 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(image, text, (x + 2, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_box(
    cv2: Any,
    image: Any,
    bbox: list[Any],
    label: str,
    color: tuple[int, int, int],
    *,
    dashed: bool,
) -> None:
    height, width = image.shape[:2]
    normalized = normalize_bbox_xyxy(bbox, width, height)
    if normalized is None:
        return
    x1, y1, x2, y2 = [int(round(value)) for value in normalized]
    if dashed:
        dash = 14
        gap = 8
        for start, end in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
            sx, sy = start
            ex, ey = end
            length = math.hypot(ex - sx, ey - sy)
            if length <= 0:
                continue
            vx = (ex - sx) / length
            vy = (ey - sy) / length
            pos = 0.0
            while pos < length:
                seg_end = min(pos + dash, length)
                p1 = (int(round(sx + vx * pos)), int(round(sy + vy * pos)))
                p2 = (int(round(sx + vx * seg_end)), int(round(sy + vy * seg_end)))
                cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
                pos += dash + gap
        draw_label(cv2, image, label, (x1, min(height - 2, y2 + 18)), color)
    else:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        draw_label(cv2, image, label, (x1, max(0, y1 - 6)), color)


def save_error_visualization(
    *,
    cv2: Any,
    dataset_root: Path,
    output_dir: Path,
    record: dict[str, Any],
    projection_record: dict[str, Any] | None,
    dino_record: dict[str, Any] | None,
) -> list[str]:
    statuses = {
        match["match_status"]
        for match in record["matches"]
        if match["match_status"] != "tp"
    }
    if record["unmatched_projection_boxes"]:
        statuses.add("unmatched_projection")
    if record["unmatched_dino_boxes"]:
        statuses.add("unmatched_dino")
    if not statuses:
        return []

    image_path = pick_rgb_image(dataset_root, projection_record, dino_record)
    if image_path is None:
        return []
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return []

    for match in record["matches"]:
        status = match["match_status"]
        color = VIS_COLORS.get(status, (220, 220, 220))
        draw_box(
            cv2,
            image,
            match["projection"].get("bbox_xyxy") or [0, 0, 0, 0],
            f"REF {status} {match['iou']:.2f}",
            color,
            dashed=False,
        )
        draw_box(
            cv2,
            image,
            match["dino"].get("bbox_xyxy") or [0, 0, 0, 0],
            f"DINO {status}",
            color,
            dashed=True,
        )

    for item in record["unmatched_projection_boxes"]:
        color = VIS_COLORS["unmatched_projection"]
        draw_box(
            cv2,
            image,
            item["projection"].get("bbox_xyxy") or [0, 0, 0, 0],
            f"REF unmatched best={item.get('best_iou')}",
            color,
            dashed=False,
        )

    for item in record["unmatched_dino_boxes"]:
        color = VIS_COLORS["unmatched_dino"]
        draw_box(
            cv2,
            image,
            item["dino"].get("bbox_xyxy") or [0, 0, 0, 0],
            f"DINO unmatched best={item.get('best_iou')}",
            color,
            dashed=True,
        )

    written = []
    pcd_timestamp = record["pcd_timestamp"]
    camera = record["camera"]
    for status in sorted(statuses):
        path = output_dir / "error_vis" / status / camera / f"{pcd_timestamp}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(path), image):
            written.append(str(path))
    return written


def main() -> None:
    args = parse_args()
    if args.low_iou_threshold < 0.0:
        raise ValueError("--low-iou-threshold must be >= 0.")
    if args.iou_threshold <= 0.0 or args.iou_threshold > 1.0:
        raise ValueError("--iou-threshold must be in (0, 1].")
    if args.low_iou_threshold > args.iou_threshold:
        raise ValueError("--low-iou-threshold must be <= --iou-threshold.")

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

    projection_records = read_jsonl(projection_jsonl)
    dino_records = read_jsonl(dino_jsonl)
    projection_by_key = index_records(projection_records)
    dino_by_key = index_records(dino_records)
    keys = sorted(set(projection_by_key) | set(dino_by_key))
    if args.cameras:
        camera_filter = set(args.cameras)
        keys = [key for key in keys if key[1] in camera_filter]
    if args.valid_time_only:
        keys = [
            key
            for key in keys
            if is_valid_time_record(projection_by_key.get(key)) or is_valid_time_record(dino_by_key.get(key))
        ]
    if args.max_records is not None:
        keys = keys[: args.max_records]

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

    manifest = {
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
            "valid_time_only": bool(args.valid_time_only),
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
            record = evaluate_record(
                dataset_root=dataset_root,
                projection=projection,
                dino=dino,
                iou_threshold=args.iou_threshold,
                low_iou_threshold=args.low_iou_threshold,
                class_aware_matching=bool(args.class_aware_matching),
                warnings=warnings,
            )
            write_jsonl_line(matches_writer, record)
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

            update_aggregate_from_record(global_aggregate, record)
            camera_aggregate = camera_aggregates[str(record["camera"])]
            update_aggregate_from_record(camera_aggregate, record)
            update_class_aggregate(class_aggregates, record)

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
        "warnings": sorted(set(warnings)),
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
    write_json(output_dir / "metrics_by_camera.json", {"schema_version": "roi_projection_dino_metrics_by_camera.v1", "by_camera": by_camera})
    write_json(output_dir / "metrics_by_class.json", {"schema_version": "roi_projection_dino_metrics_by_class.v1", "by_class": by_class})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
