"""Camera image rendering for 3D box projections.

This module keeps the pixel-level rendering workflow shared by the plain
projection script and the ROI-aware variant. Passing ``roi_config=None`` keeps
the original non-ROI return shape; passing a ``RoiConfig`` enables ROI metadata,
ROI counters, optional ROI filtering, and optional debug visualization output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.calibration import CameraCalibration
from utils.common import array_to_rounded_list, round_float
from utils.detection import DetectionBox
from utils.geometry import (
    bbox_xyxy_from_points,
    box_to_corners,
    clip_bbox_xyxy,
    project_points,
    projected_bbox_intersects,
    transform_points,
)
from utils.labels import CLASS_COLORS, DEFAULT_COLOR, canonicalize_label
from utils.roi import RoiConfig, build_roi_info, empty_roi_counts
from utils.visualization import draw_box_edges


DEBUG_ROI_IN_COLOR = (60, 220, 60)
DEBUG_ROI_OUT_COLOR = (40, 40, 255)
DEBUG_ROI_UNKNOWN_COLOR = (170, 170, 170)


def debug_color_for_roi(roi: dict[str, Any]) -> tuple[int, int, int]:
    if roi.get("in_roi") is True:
        return DEBUG_ROI_IN_COLOR
    if roi.get("in_roi") is False:
        return DEBUG_ROI_OUT_COLOR
    return DEBUG_ROI_UNKNOWN_COLOR


def format_roi_position(roi: dict[str, Any]) -> str:
    longitudinal = roi.get("longitudinal_m")
    lateral = roi.get("lateral_m")
    if isinstance(longitudinal, (int, float)) and isinstance(lateral, (int, float)):
        return f"long={float(longitudinal):.2f}m lat={float(lateral):.2f}m"
    return "long=na lat=na"


def debug_label_for_roi(class_name: str, score: float, roi: dict[str, Any]) -> str:
    if roi.get("in_roi") is True:
        status = "roi"
    elif roi.get("in_roi") is False:
        status = "out"
    else:
        status = "unknown"
    return f"{class_name} {score:.2f} {status} {format_roi_position(roi)}"


def render_camera_image(
    *,
    image_path: Path,
    output_path: Path,
    output_by_image_path: Path,
    boxes: list[DetectionBox],
    calibration: CameraCalibration,
    t_box_to_ego: np.ndarray,
    min_depth: float,
    pcd_timestamp: int,
    camera_name: str,
    roi_config: RoiConfig | None = None,
    debug_output_path: Path | None = None,
) -> dict[str, Any]:
    """Render 3D detection boxes on a camera image.

    Args:
        image_path: Input camera image path.
        output_path: Output path keyed by PCD timestamp.
        output_by_image_path: Output path keyed by image timestamp.
        boxes: Detection boxes already filtered by score threshold.
        calibration: Camera calibration used by the projection step.
        t_box_to_ego: 4x4 transform from box source frame to ego frame.
        min_depth: Minimum valid camera-frame depth for all box corners.
        pcd_timestamp: PCD frame timestamp, used to build projection ids.
        camera_name: Camera stream name, used to build projection ids.
        roi_config: Optional ROI config. When omitted, return shape matches the
            non-ROI projection script.
        debug_output_path: Optional ROI debug image output path. It is only used
            when roi_config is provided.

    Returns:
        A result dict containing image paths, counters, projected boxes, and
        image dimensions. ROI fields are present only when roi_config is given.
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    roi_enabled_for_output = roi_config is not None
    debug_image = (
        image.copy()
        if roi_enabled_for_output and debug_output_path is not None
        else None
    )

    projected_count = 0
    skipped_behind = 0
    skipped_outside = 0
    skipped_unknown = 0
    roi_counts = empty_roi_counts() if roi_enabled_for_output else {}
    projected_boxes: list[dict[str, Any]] = []

    for box in boxes:
        class_name, class_id = canonicalize_label(box.category)
        if class_name is None or class_id is None:
            skipped_unknown += 1
            continue

        corners_box = box_to_corners(box.center, box.size_lwh, box.heading)
        corners_ego = transform_points(corners_box, t_box_to_ego)
        center_ego = None
        roi: dict[str, Any] | None = None

        if roi_config is not None:
            center_ego = transform_points(box.center.reshape(1, 3), t_box_to_ego)[0]
            roi = build_roi_info(center_ego, roi_config)
            if roi_config.enabled:
                if roi["in_roi"]:
                    roi_counts["boxes_in_roi"] += 1
                else:
                    roi_counts["boxes_outside_roi"] += 1
                    if roi_config.filter_enabled:
                        roi_counts["boxes_skipped_outside_roi"] += 1
                        if debug_image is None:
                            continue

        corners_cam = transform_points(corners_ego, calibration.t_ego_to_cam)
        if np.any(corners_cam[:, 2] <= min_depth):
            skipped_behind += 1
            continue

        corners_2d = project_points(corners_cam, calibration)
        if not projected_bbox_intersects(corners_2d, calibration.width, calibration.height):
            skipped_outside += 1
            continue

        if debug_image is not None and roi is not None:
            debug_color = debug_color_for_roi(roi)
            debug_label = debug_label_for_roi(class_name, box.score, roi)
            draw_box_edges(debug_image, corners_2d, debug_label, debug_color)

        if (
            roi_config is not None
            and roi_config.enabled
            and roi_config.filter_enabled
            and roi is not None
            and roi.get("in_roi") is False
        ):
            continue

        bbox_unclipped = bbox_xyxy_from_points(corners_2d)
        bbox_clipped = clip_bbox_xyxy(
            bbox_unclipped,
            image.shape[1],
            image.shape[0],
        )
        projection_id = f"projection:{pcd_timestamp}:{camera_name}:{projected_count}"

        if roi_config is not None and roi_config.enabled and roi is not None:
            if roi["in_roi"]:
                roi_counts["boxes_projected_in_roi"] += 1
            else:
                roi_counts["boxes_projected_outside_roi"] += 1

        if roi_config is not None and center_ego is not None:
            bbox_3d = {
                "center": array_to_rounded_list(box.center),
                "center_ego": array_to_rounded_list(center_ego),
                "size_lwh": array_to_rounded_list(box.size_lwh),
                "heading": round_float(box.heading),
            }
        else:
            bbox_3d = {
                "center": array_to_rounded_list(box.center),
                "size_lwh": array_to_rounded_list(box.size_lwh),
                "heading": round_float(box.heading),
            }

        if roi_config is not None:
            projected_box = {
                "projection_id": projection_id,
                "object_id": box.object_id,
                "class_name": class_name,
                "class_id": class_id,
                "label_raw": box.category,
                "score": round_float(box.score),
                "bbox_xyxy": bbox_clipped,
                "bbox_xyxy_unclipped": bbox_unclipped,
                "corners_2d": array_to_rounded_list(corners_2d),
                "corners_3d_camera": array_to_rounded_list(corners_cam),
                "depth_range": {
                    "min": round_float(np.min(corners_cam[:, 2])),
                    "max": round_float(np.max(corners_cam[:, 2])),
                },
                "bbox_3d": bbox_3d,
                "roi": roi,
                "visibility": {
                    "projected": True,
                    "skip_reason": None,
                },
                "extra": {},
            }
        else:
            projected_box = {
                "projection_id": projection_id,
                "object_id": box.object_id,
                "class_name": class_name,
                "class_id": class_id,
                "label_raw": box.category,
                "score": round_float(box.score),
                "bbox_xyxy": bbox_clipped,
                "bbox_xyxy_unclipped": bbox_unclipped,
                "corners_2d": array_to_rounded_list(corners_2d),
                "corners_3d_camera": array_to_rounded_list(corners_cam),
                "depth_range": {
                    "min": round_float(np.min(corners_cam[:, 2])),
                    "max": round_float(np.max(corners_cam[:, 2])),
                },
                "bbox_3d": bbox_3d,
                "visibility": {
                    "projected": True,
                    "skip_reason": None,
                },
                "extra": {},
            }

        projected_boxes.append(projected_box)

        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
        label = f"{class_name} {box.score:.2f}"
        draw_box_edges(image, corners_2d, label, color)
        projected_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise RuntimeError(f"failed to write image: {output_path}")

    output_by_image_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_by_image_path), image)
    if not ok:
        raise RuntimeError(f"failed to write image: {output_by_image_path}")

    if debug_output_path is not None and debug_image is not None:
        debug_output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(debug_output_path), debug_image)
        if not ok:
            raise RuntimeError(f"failed to write image: {debug_output_path}")

    if roi_enabled_for_output:
        return {
            "input_image": str(image_path),
            "output_image": str(output_path),
            "output_image_by_image": str(output_by_image_path),
            "debug_output_image": str(debug_output_path) if debug_output_path else None,
            "boxes_total": len(boxes),
            "boxes_projected": projected_count,
            "boxes_skipped_behind_camera": skipped_behind,
            "boxes_skipped_outside_image": skipped_outside,
            "boxes_skipped_unknown_class": skipped_unknown,
            **roi_counts,
            "projected_boxes": projected_boxes,
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
        }

    return {
        "input_image": str(image_path),
        "output_image": str(output_path),
        "output_image_by_image": str(output_by_image_path),
        "boxes_total": len(boxes),
        "boxes_projected": projected_count,
        "boxes_skipped_behind_camera": skipped_behind,
        "boxes_skipped_outside_image": skipped_outside,
        "boxes_skipped_unknown_class": skipped_unknown,
        "projected_boxes": projected_boxes,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
    }
