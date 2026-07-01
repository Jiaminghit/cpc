"""BEV rendering for 3D detection boxes and point clouds."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.common import array_to_rounded_list, round_float
from utils.detection import DetectionBox
from utils.geometry import box_to_corners, transform_points
from utils.labels import CLASS_COLORS, DEFAULT_COLOR, canonicalize_label
from utils.pcd import read_binary_pcd_xyzi
from utils.roi import RoiConfig, build_roi_info, empty_roi_counts


BOTTOM_CORNER_INDICES = [0, 1, 2, 3]
BACKGROUND_COLOR = (24, 26, 28)
GRID_COLOR = (55, 58, 62)
AXIS_COLOR = (160, 160, 160)
EGO_COLOR = (245, 245, 245)
ROI_COLOR = (90, 180, 255)
TEXT_COLOR = (230, 230, 230)


@dataclass(frozen=True)
class BevConfig:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    meters_per_pixel: float
    width: int
    height: int
    y_positive_direction: str
    draw_labels: bool
    draw_heading: bool
    grid_interval_m: float


@dataclass(frozen=True)
class PointCloudConfig:
    enabled: bool
    source_frame: str
    color_mode: str
    z_min: float
    z_max: float
    point_size: int


def ego_to_bev_pixels(points_xy: np.ndarray, bev_config: BevConfig) -> np.ndarray:
    x_values = points_xy[:, 0]
    y_values = points_xy[:, 1]
    if bev_config.y_positive_direction == "left":
        u_values = (bev_config.y_max - y_values) / (bev_config.y_max - bev_config.y_min) * (bev_config.width - 1)
    else:
        u_values = (y_values - bev_config.y_min) / (bev_config.y_max - bev_config.y_min) * (bev_config.width - 1)
    v_values = (bev_config.x_max - x_values) / (bev_config.x_max - bev_config.x_min) * (bev_config.height - 1)
    return np.stack([u_values, v_values], axis=1)


def bev_bbox_intersects(points_px: np.ndarray, width: int, height: int) -> bool:
    min_x = float(np.min(points_px[:, 0]))
    max_x = float(np.max(points_px[:, 0]))
    min_y = float(np.min(points_px[:, 1]))
    max_y = float(np.max(points_px[:, 1]))
    return max_x >= 0 and max_y >= 0 and min_x < width and min_y < height


def draw_line_if_visible(
    image: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    height, width = image.shape[:2]
    ok, clipped_p1, clipped_p2 = cv2.clipLine((0, 0, width, height), p1, p2)
    if ok:
        cv2.line(image, clipped_p1, clipped_p2, color, thickness, cv2.LINE_AA)


def build_bev_canvas(bev_config: BevConfig, roi_config: RoiConfig) -> np.ndarray:
    canvas = np.full((bev_config.height, bev_config.width, 3), BACKGROUND_COLOR, dtype=np.uint8)

    first_x = math.ceil(bev_config.x_min / bev_config.grid_interval_m) * bev_config.grid_interval_m
    x = first_x
    while x <= bev_config.x_max:
        p1, p2 = ego_to_bev_pixels(
            np.asarray([[x, bev_config.y_min], [x, bev_config.y_max]], dtype=np.float64),
            bev_config,
        )
        color = AXIS_COLOR if abs(x) < 1e-6 else GRID_COLOR
        draw_line_if_visible(canvas, tuple(np.rint(p1).astype(int)), tuple(np.rint(p2).astype(int)), color)
        x += bev_config.grid_interval_m

    first_y = math.ceil(bev_config.y_min / bev_config.grid_interval_m) * bev_config.grid_interval_m
    y = first_y
    while y <= bev_config.y_max:
        p1, p2 = ego_to_bev_pixels(
            np.asarray([[bev_config.x_min, y], [bev_config.x_max, y]], dtype=np.float64),
            bev_config,
        )
        color = AXIS_COLOR if abs(y) < 1e-6 else GRID_COLOR
        draw_line_if_visible(canvas, tuple(np.rint(p1).astype(int)), tuple(np.rint(p2).astype(int)), color)
        y += bev_config.grid_interval_m

    return canvas


def colors_for_points(
    points_ego: np.ndarray,
    intensity: np.ndarray | None,
    point_config: PointCloudConfig,
) -> np.ndarray:
    if point_config.color_mode == "constant":
        colors = np.full((points_ego.shape[0], 3), (82, 82, 82), dtype=np.uint8)
        return colors

    if point_config.color_mode == "intensity" and intensity is not None:
        values = intensity.astype(np.float64)
        finite = np.isfinite(values)
        if np.any(finite):
            lo = float(np.percentile(values[finite], 2))
            hi = float(np.percentile(values[finite], 98))
            denom = max(hi - lo, 1e-6)
            normalized = np.clip((values - lo) / denom, 0.0, 1.0)
        else:
            normalized = np.zeros(values.shape, dtype=np.float64)
    else:
        denom = max(point_config.z_max - point_config.z_min, 1e-6)
        normalized = np.clip((points_ego[:, 2] - point_config.z_min) / denom, 0.0, 1.0)

    gray = (55 + normalized * 180).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=1)


def draw_point_cloud_on_bev(
    canvas: np.ndarray,
    *,
    pcd_path: Path | None,
    t_pcd_to_ego: np.ndarray,
    bev_config: BevConfig,
    point_config: PointCloudConfig,
) -> dict[str, Any]:
    stats = {
        "enabled": point_config.enabled,
        "pcd_path": str(pcd_path) if pcd_path else None,
        "source_frame": point_config.source_frame,
        "points_total": 0,
        "points_valid": 0,
        "points_drawn": 0,
        "skipped": False,
        "reason": None,
    }
    if not point_config.enabled:
        stats["skipped"] = True
        stats["reason"] = "disabled"
        return stats
    if pcd_path is None:
        stats["skipped"] = True
        stats["reason"] = "pcd_path_missing"
        return stats
    if not pcd_path.is_file():
        stats["skipped"] = True
        stats["reason"] = "pcd_file_missing"
        return stats

    points, intensity = read_binary_pcd_xyzi(pcd_path)
    stats["points_total"] = int(points.shape[0])
    finite_mask = np.all(np.isfinite(points), axis=1)
    nonzero_mask = np.any(np.abs(points[:, :3]) > 1e-6, axis=1)
    valid_mask = finite_mask & nonzero_mask
    points = points[valid_mask]
    if intensity is not None:
        intensity = intensity[valid_mask]

    if points.size == 0:
        stats["points_valid"] = 0
        stats["points_drawn"] = 0
        return stats

    points_ego = transform_points(points, t_pcd_to_ego)
    range_mask = (
        np.isfinite(points_ego[:, 0])
        & np.isfinite(points_ego[:, 1])
        & np.isfinite(points_ego[:, 2])
        & (points_ego[:, 0] >= bev_config.x_min)
        & (points_ego[:, 0] <= bev_config.x_max)
        & (points_ego[:, 1] >= bev_config.y_min)
        & (points_ego[:, 1] <= bev_config.y_max)
        & (points_ego[:, 2] >= point_config.z_min)
        & (points_ego[:, 2] <= point_config.z_max)
    )
    points_ego = points_ego[range_mask]
    if intensity is not None:
        intensity = intensity[range_mask]
    stats["points_valid"] = int(points_ego.shape[0])

    if points_ego.size == 0:
        return stats

    points_px = np.rint(ego_to_bev_pixels(points_ego[:, :2], bev_config)).astype(np.int32)
    in_canvas = (
        (points_px[:, 0] >= 0)
        & (points_px[:, 0] < bev_config.width)
        & (points_px[:, 1] >= 0)
        & (points_px[:, 1] < bev_config.height)
    )
    points_px = points_px[in_canvas]
    points_ego = points_ego[in_canvas]
    if intensity is not None:
        intensity = intensity[in_canvas]
    stats["points_drawn"] = int(points_px.shape[0])

    if points_px.size == 0:
        return stats

    colors = colors_for_points(points_ego, intensity, point_config)
    y_indices = points_px[:, 1]
    x_indices = points_px[:, 0]
    if point_config.point_size == 1:
        canvas[y_indices, x_indices] = np.maximum(canvas[y_indices, x_indices], colors)
    else:
        radius = max(1, point_config.point_size // 2)
        for point, color in zip(points_px, colors):
            cv2.circle(canvas, tuple(point.tolist()), radius, tuple(int(v) for v in color.tolist()), -1, cv2.LINE_AA)
    return stats


def draw_ego_marker(canvas: np.ndarray, bev_config: BevConfig) -> None:
    origin = ego_to_bev_pixels(np.asarray([[0.0, 0.0]], dtype=np.float64), bev_config)[0]
    forward = ego_to_bev_pixels(np.asarray([[10.0, 0.0]], dtype=np.float64), bev_config)[0]
    origin_pt = tuple(np.rint(origin).astype(int))
    forward_pt = tuple(np.rint(forward).astype(int))
    if 0 <= origin_pt[0] < bev_config.width and 0 <= origin_pt[1] < bev_config.height:
        cv2.circle(canvas, origin_pt, 5, EGO_COLOR, -1, cv2.LINE_AA)
    draw_line_if_visible(canvas, origin_pt, forward_pt, EGO_COLOR, 2)
    cv2.putText(canvas, "ego", (max(origin_pt[0] + 8, 0), max(origin_pt[1] - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)


def draw_roi_rectangle(canvas: np.ndarray, bev_config: BevConfig, roi_config: RoiConfig) -> None:
    if not roi_config.enabled:
        return
    if {roi_config.longitudinal_axis, roi_config.lateral_axis} != {"x", "y"}:
        return

    axis_ranges = {
        roi_config.longitudinal_axis: roi_config.longitudinal_range_m,
        roi_config.lateral_axis: roi_config.lateral_range_m,
    }
    x_min, x_max = axis_ranges["x"]
    y_min, y_max = axis_ranges["y"]
    corners = np.asarray(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=np.float64,
    )
    points_px = np.rint(ego_to_bev_pixels(corners, bev_config)).astype(np.int32)
    cv2.polylines(canvas, [points_px], isClosed=True, color=ROI_COLOR, thickness=2, lineType=cv2.LINE_AA)


def dim_color(color: tuple[int, int, int], factor: float = 0.45) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, channel * factor))) for channel in color)


def draw_heading_arrow(
    canvas: np.ndarray,
    corners_px: np.ndarray,
    center_px: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    front_mid = (corners_px[0] + corners_px[1]) / 2.0
    start = tuple(np.rint(center_px).astype(int))
    end = tuple(np.rint(front_mid).astype(int))
    draw_line_if_visible(canvas, start, end, color, 2)


def render_bev_frame(
    *,
    output_path: Path,
    boxes: list[DetectionBox],
    t_box_to_ego: np.ndarray,
    pcd_path: Path | None,
    t_pcd_to_ego: np.ndarray,
    pcd_timestamp: int,
    bev_config: BevConfig,
    roi_config: RoiConfig,
    point_config: PointCloudConfig,
) -> dict[str, Any]:
    image = build_bev_canvas(bev_config, roi_config)
    point_cloud_stats = draw_point_cloud_on_bev(
        image,
        pcd_path=pcd_path,
        t_pcd_to_ego=t_pcd_to_ego,
        bev_config=bev_config,
        point_config=point_config,
    )
    draw_roi_rectangle(image, bev_config, roi_config)
    draw_ego_marker(image, bev_config)

    boxes_drawn = 0
    skipped_unknown = 0
    skipped_outside_canvas = 0
    roi_counts = empty_roi_counts()
    rendered_boxes: list[dict[str, Any]] = []

    for box in boxes:
        class_name, class_id = canonicalize_label(box.category)
        if class_name is None or class_id is None:
            skipped_unknown += 1
            continue

        corners_box = box_to_corners(box.center, box.size_lwh, box.heading)
        corners_ego = transform_points(corners_box, t_box_to_ego)
        center_ego = transform_points(box.center.reshape(1, 3), t_box_to_ego)[0]
        roi = build_roi_info(center_ego, roi_config)

        if roi_config.enabled:
            if roi["in_roi"]:
                roi_counts["boxes_in_roi"] += 1
            else:
                roi_counts["boxes_outside_roi"] += 1
                if roi_config.filter_enabled:
                    roi_counts["boxes_skipped_outside_roi"] += 1
                    continue

        bottom_corners_ego = corners_ego[BOTTOM_CORNER_INDICES][:, :2]
        polygon_px = ego_to_bev_pixels(bottom_corners_ego, bev_config)
        if not bev_bbox_intersects(polygon_px, bev_config.width, bev_config.height):
            skipped_outside_canvas += 1
            continue

        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
        if roi_config.enabled and roi.get("in_roi") is False:
            color = dim_color(color)

        polygon_px_int = np.rint(polygon_px).astype(np.int32)
        cv2.polylines(image, [polygon_px_int], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        center_px = ego_to_bev_pixels(center_ego[:2].reshape(1, 2), bev_config)[0]
        if bev_config.draw_heading:
            draw_heading_arrow(image, polygon_px, center_px, color)

        if bev_config.draw_labels:
            label = f"{class_name} {box.score:.2f}"
            label_xy = tuple(np.rint(np.min(polygon_px, axis=0)).astype(int))
            label_x = int(np.clip(label_xy[0], 0, bev_config.width - 1))
            label_y = int(np.clip(label_xy[1] - 5, 12, bev_config.height - 1))
            cv2.putText(image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        if roi_config.enabled:
            if roi["in_roi"]:
                roi_counts["boxes_projected_in_roi"] += 1
            else:
                roi_counts["boxes_projected_outside_roi"] += 1

        rendered_boxes.append(
            {
                "bev_id": f"bev:{pcd_timestamp}:{boxes_drawn}",
                "object_id": box.object_id,
                "class_name": class_name,
                "class_id": class_id,
                "label_raw": box.category,
                "score": round_float(box.score),
                "bbox_3d": {
                    "center": array_to_rounded_list(box.center),
                    "center_ego": array_to_rounded_list(center_ego),
                    "size_lwh": array_to_rounded_list(box.size_lwh),
                    "heading": round_float(box.heading),
                },
                "bev_polygon_px": array_to_rounded_list(polygon_px),
                "bev_polygon_ego": array_to_rounded_list(bottom_corners_ego),
                "roi": roi,
                "visibility": {
                    "projected": True,
                    "skip_reason": None,
                },
            }
        )
        boxes_drawn += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise RuntimeError(f"failed to write BEV image: {output_path}")

    return {
        "bev_image": str(output_path),
        "boxes_total": len(boxes),
        "boxes_drawn": boxes_drawn,
        "boxes_skipped_unknown_class": skipped_unknown,
        "boxes_skipped_outside_canvas": skipped_outside_canvas,
        **roi_counts,
        "boxes": rendered_boxes,
        "point_cloud": point_cloud_stats,
        "bev_width": bev_config.width,
        "bev_height": bev_config.height,
    }


def bev_config_to_dict(bev_config: BevConfig) -> dict[str, Any]:
    return {
        "x_range_m": [round_float(bev_config.x_min), round_float(bev_config.x_max)],
        "y_range_m": [round_float(bev_config.y_min), round_float(bev_config.y_max)],
        "meters_per_pixel": round_float(bev_config.meters_per_pixel),
        "width": bev_config.width,
        "height": bev_config.height,
        "y_positive_direction": bev_config.y_positive_direction,
        "draw_labels": bev_config.draw_labels,
        "draw_heading": bev_config.draw_heading,
        "grid_interval_m": round_float(bev_config.grid_interval_m),
    }


def point_cloud_config_to_dict(point_config: PointCloudConfig) -> dict[str, Any]:
    return {
        "enabled": point_config.enabled,
        "source_frame": point_config.source_frame,
        "color_mode": point_config.color_mode,
        "z_range_m": [round_float(point_config.z_min), round_float(point_config.z_max)],
        "point_size": point_config.point_size,
    }
