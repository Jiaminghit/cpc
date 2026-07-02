#!/usr/bin/env python3
"""Overlay projected 3D reference boxes and aligned model detections.

This is stage 3A of the projection/model comparison workflow. It only creates
visual overlays and summary metadata; it does not compute IoU or TP/FP/FN.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CLASS_TABLE = [
    {"class_id": 0, "class_name": "Car"},
    {"class_id": 1, "class_name": "Pedestrian"},
    {"class_id": 2, "class_name": "Cyclist"},
    {"class_id": 3, "class_name": "Van"},
    {"class_id": 4, "class_name": "Traffic_cone"},
]

CLASS_COLORS = {
    "Car": (255, 80, 40),
    "Van": (255, 220, 40),
    "Pedestrian": (60, 220, 60),
    "Cyclist": (40, 220, 255),
    "Traffic_cone": (40, 40, 255),
}
DEFAULT_COLOR = (220, 220, 220)
DEFAULT_MODEL_MASK_ALPHA = 0.16

BOX_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay projection reference boxes and model detections."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root directory.")
    parser.add_argument("--projection-jsonl", required=True, help="projections_aligned.jsonl path.")
    parser.add_argument("--model-jsonl", required=True, help="detections_aligned.jsonl path.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <dataset-root>/compare_grounding_dino_b_projection.",
    )
    parser.add_argument("--save-vis", action="store_true", help="Write overlay images.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional max aligned records.")
    parser.add_argument("--cameras", nargs="+", default=None, help="Optional camera subset.")
    parser.add_argument(
        "--draw-projection-corners",
        action="store_true",
        help="Compatibility flag. Reference boxes are now always drawn from corners_2d.",
    )
    parser.add_argument(
        "--model-name",
        default="grounding_dino_b",
        help="Model name recorded in manifest/summary.",
    )
    parser.add_argument(
        "--model-mask-alpha",
        type=float,
        default=DEFAULT_MODEL_MASK_ALPHA,
        help="Transparent fill alpha for model detection boxes.",
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
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def aligned_key(record: dict[str, Any]) -> tuple[int, str]:
    return int(record["pcd_timestamp"]), str(record["camera"])


def resolve_path(dataset_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == dataset_root.name:
        return dataset_root.parent / path
    return dataset_root / path


def pick_rgb_image(dataset_root: Path, projection: dict[str, Any] | None, model: dict[str, Any] | None) -> Path | None:
    for record in (projection, model):
        if not record:
            continue
        image = resolve_path(dataset_root, record.get("rgb_image"))
        if image and image.is_file():
            return image
        image_rel = resolve_path(dataset_root, record.get("rgb_image_rel"))
        if image_rel and image_rel.is_file():
            return image_rel
    return None


def color_for(class_name: str | None) -> tuple[int, int, int]:
    if not class_name:
        return DEFAULT_COLOR
    return CLASS_COLORS.get(class_name, DEFAULT_COLOR)


def contrast_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(255 - int(channel) for channel in color)


def clean_bbox(bbox: list[Any], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = int(round(np.clip(x1, 0, width - 1)))
    y1 = int(round(np.clip(y1, 0, height - 1)))
    x2 = int(round(np.clip(x2, 0, width - 1)))
    y2 = int(round(np.clip(y2, 0, height - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = int(np.clip(origin[0], 0, max(0, width - text_w - 2)))
    y = int(np.clip(origin[1], text_h + baseline + 2, height - 2))
    cv2.rectangle(
        image,
        (x, y - text_h - baseline - 3),
        (x + text_w + 4, y + baseline + 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(image, text, (x + 2, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_solid_box(
    image: np.ndarray,
    bbox: list[Any],
    label: str,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clean_bbox(bbox, width, height)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    draw_label(image, label, (x1, max(0, y1 - 6)), color)


def draw_dashed_line(
    image: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash_length: int = 14,
    gap_length: int = 8,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    distance = float(np.hypot(dx, dy))
    if distance <= 0:
        return
    vx = dx / distance
    vy = dy / distance
    pos = 0.0
    while pos < distance:
        end = min(pos + dash_length, distance)
        start_pt = (int(round(x1 + vx * pos)), int(round(y1 + vy * pos)))
        end_pt = (int(round(x1 + vx * end)), int(round(y1 + vy * end)))
        cv2.line(image, start_pt, end_pt, color, thickness, cv2.LINE_AA)
        pos += dash_length + gap_length


def draw_dashed_box(
    image: np.ndarray,
    bbox: list[Any],
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
    mask_alpha: float = DEFAULT_MODEL_MASK_ALPHA,
) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clean_bbox(bbox, width, height)
    if mask_alpha > 0 and x2 > x1 and y2 > y1:
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(
            overlay,
            float(np.clip(mask_alpha, 0.0, 1.0)),
            image,
            1.0 - float(np.clip(mask_alpha, 0.0, 1.0)),
            0,
            dst=image,
        )
    draw_dashed_line(image, (x1, y1), (x2, y1), color, thickness)
    draw_dashed_line(image, (x2, y1), (x2, y2), color, thickness)
    draw_dashed_line(image, (x2, y2), (x1, y2), color, thickness)
    draw_dashed_line(image, (x1, y2), (x1, y1), color, thickness)
    draw_label(image, label, (x1, min(height - 2, y2 + 18)), color)


def draw_corner_edges(
    image: np.ndarray,
    corners_2d: list[list[Any]],
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    if len(corners_2d) != 8:
        return
    height, width = image.shape[:2]
    rect = (0, 0, width, height)
    points = np.rint(np.asarray(corners_2d, dtype=np.float64)).astype(np.int32)
    for start, end in BOX_EDGES:
        p1 = tuple(points[start].tolist())
        p2 = tuple(points[end].tolist())
        ok, clipped_p1, clipped_p2 = cv2.clipLine(rect, p1, p2)
        if ok:
            cv2.line(image, clipped_p1, clipped_p2, color, thickness, cv2.LINE_AA)
    label_x = int(np.clip(np.min(points[:, 0]), 0, width - 1))
    label_y = int(np.clip(np.min(points[:, 1]) - 6, 16, height - 1))
    draw_label(image, label, (label_x, label_y), color)


def box_label(prefix: str, box: dict[str, Any]) -> str:
    class_name = box.get("class_name") or box.get("label_raw") or "unknown"
    score = box.get("score")
    if score is None:
        return f"{prefix} {class_name}"
    return f"{prefix} {class_name} {float(score):.2f}"


def overlay_record(
    *,
    dataset_root: Path,
    projection: dict[str, Any] | None,
    model: dict[str, Any] | None,
    output_path: Path,
    save_vis: bool,
    draw_projection_corners: bool,
    model_mask_alpha: float,
) -> dict[str, Any]:
    base_record = projection or model
    if base_record is None:
        raise ValueError("projection and model cannot both be None")

    pcd_timestamp, camera = aligned_key(base_record)
    image_path = pick_rgb_image(dataset_root, projection, model)
    reference_boxes = projection.get("projected_boxes", []) if projection else []
    model_boxes = model.get("detections", []) if model else []

    status = "ok"
    reason = None
    if image_path is None:
        status = "skipped"
        reason = "rgb_image_missing"
    elif save_vis:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            status = "skipped"
            reason = "rgb_image_read_failed"
        else:
            for box in reference_boxes:
                color = color_for(box.get("class_name"))
                draw_corner_edges(
                    image,
                    box.get("corners_2d", []),
                    box_label("REF", box),
                    color,
                    thickness=2,
                )
            for box in model_boxes:
                color = contrast_color(color_for(box.get("class_name")))
                draw_dashed_box(
                    image,
                    box.get("bbox_xyxy", [0, 0, 0, 0]),
                    box_label("DINO", box),
                    color,
                    thickness=2,
                    mask_alpha=model_mask_alpha,
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(output_path), image)
            if not ok:
                status = "skipped"
                reason = "overlay_write_failed"

    return {
        "pcd_timestamp": pcd_timestamp,
        "camera": camera,
        "rgb_timestamp": base_record.get("rgb_timestamp"),
        "rgb_image": str(image_path) if image_path else None,
        "overlay_image": str(output_path) if status == "ok" and save_vis else None,
        "status": status,
        "reason": reason,
        "reference_box_count": len(reference_boxes),
        "model_box_count": len(model_boxes),
        "has_projection_record": projection is not None,
        "has_model_record": model is not None,
    }


def index_records(records: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for record in records:
        indexed[aligned_key(record)] = record
    return indexed


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    projection_jsonl = Path(args.projection_jsonl).expanduser().resolve()
    model_jsonl = Path(args.model_jsonl).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else dataset_root / "compare_grounding_dino_b_projection"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    projection_records = read_jsonl(projection_jsonl)
    model_records = read_jsonl(model_jsonl)
    projection_by_key = index_records(projection_records)
    model_by_key = index_records(model_records)
    keys = sorted(set(projection_by_key) | set(model_by_key))
    if args.cameras:
        cameras = set(args.cameras)
        keys = [key for key in keys if key[1] in cameras]
    if args.max_records is not None:
        keys = keys[: args.max_records]

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

    summary = {
        "schema_version": "overlay_summary.v1",
        "stage": "3A_visual_overlay_only",
        "num_projection_records": len(projection_records),
        "num_model_records": len(model_records),
        "num_selected_records": len(keys),
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
        "note": "No IoU, matching, TP/FP/FN, precision, or recall is computed in this stage.",
    }
    write_json(output_dir / "overlay_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
