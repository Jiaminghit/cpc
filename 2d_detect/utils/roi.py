from __future__ import annotations

"""LiDAR ROI 判断与 bbox 内点选择逻辑。

核心思路：把投影到图像的 LiDAR 点落入 2D bbox 的点找出来，
取靠近相机的一部分点，使用其中位数作为该 2D 检测框的 ego 代表点。
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.geometry import ProjectedPointCloud
from utils.io import array_to_rounded_list, round_float


DEFAULT_BOX_SOURCE_FRAME = "lidar_top_GT"
ROI_FRAME = "ego"
ROI_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
METHOD_NAME = "project_lidar_points_to_image_bbox_depth_percentile"


@dataclass(frozen=True)
class RoiConfig:
    """描述 ego ROI 使用的坐标轴和范围。"""

    enabled: bool
    frame: str
    lateral_axis: str
    longitudinal_axis: str
    lateral_range_m: tuple[float, float]
    longitudinal_range_m: tuple[float, float]


@dataclass(frozen=True)
class DetectionPointSelection:
    """保存一个 bbox 内 LiDAR 点选择过程中的调试数据。"""

    points_in_bbox: int
    points_used: int
    in_bbox_pixels: np.ndarray
    used_pixels: np.ndarray
    used_camera: np.ndarray
    used_ego: np.ndarray


def roi_config_to_dict(roi_config: RoiConfig) -> dict[str, Any]:
    """将 ROI 配置转换为可写入 JSON 的普通 dict。"""
    return {
        "enabled": roi_config.enabled,
        "frame": roi_config.frame,
        "lateral_axis": roi_config.lateral_axis,
        "longitudinal_axis": roi_config.longitudinal_axis,
        "lateral_range_m": [
            round_float(roi_config.lateral_range_m[0]),
            round_float(roi_config.lateral_range_m[1]),
        ],
        "longitudinal_range_m": [
            round_float(roi_config.longitudinal_range_m[0]),
            round_float(roi_config.longitudinal_range_m[1]),
        ],
    }


def unknown_roi(reason: str, *, points_in_bbox: int = 0, points_used: int = 0) -> dict[str, Any]:
    """生成无法判断 ROI 时的占位信息，并记录原因。"""
    return {
        "enabled": True,
        "in_roi": None,
        "reason": reason,
        "frame": ROI_FRAME,
        "center_ego": None,
        "representative_point_ego": None,
        "representative_point_camera": None,
        "depth_camera_m": None,
        "points_in_bbox": int(points_in_bbox),
        "points_used": int(points_used),
        "method": METHOD_NAME,
    }


def build_roi_info(
    *,
    representative_point_ego: np.ndarray,
    representative_point_camera: np.ndarray,
    points_in_bbox: int,
    points_used: int,
    roi_config: RoiConfig,
) -> dict[str, Any]:
    """根据 ego 代表点判断是否落在 ROI 范围内，并生成输出字段。"""
    lateral_idx = ROI_AXIS_INDEX[roi_config.lateral_axis]
    longitudinal_idx = ROI_AXIS_INDEX[roi_config.longitudinal_axis]
    lateral = float(representative_point_ego[lateral_idx])
    longitudinal = float(representative_point_ego[longitudinal_idx])
    lateral_min, lateral_max = roi_config.lateral_range_m
    longitudinal_min, longitudinal_max = roi_config.longitudinal_range_m
    in_roi = (
        lateral_min <= lateral <= lateral_max
        and longitudinal_min <= longitudinal <= longitudinal_max
    )
    return {
        "enabled": True,
        "in_roi": bool(in_roi),
        "reason": None,
        "frame": roi_config.frame,
        "center_ego": array_to_rounded_list(representative_point_ego),
        "representative_point_ego": array_to_rounded_list(representative_point_ego),
        "representative_point_camera": array_to_rounded_list(representative_point_camera),
        "depth_camera_m": round_float(representative_point_camera[2]),
        "points_in_bbox": int(points_in_bbox),
        "points_used": int(points_used),
        "lateral_axis": roi_config.lateral_axis,
        "longitudinal_axis": roi_config.longitudinal_axis,
        "lateral_m": round_float(lateral),
        "longitudinal_m": round_float(longitudinal),
        "lateral_range_m": [round_float(lateral_min), round_float(lateral_max)],
        "longitudinal_range_m": [round_float(longitudinal_min), round_float(longitudinal_max)],
        "method": METHOD_NAME,
    }


def empty_selection() -> DetectionPointSelection:
    """构造空的点选择结果，供异常/unknown 分支复用。"""
    empty = np.empty((0, 2), dtype=np.float64)
    empty3 = np.empty((0, 3), dtype=np.float64)
    return DetectionPointSelection(0, 0, empty, empty, empty3, empty3)


def estimate_detection_position(
    *,
    detection: dict[str, Any],
    projected: ProjectedPointCloud,
    roi_config: RoiConfig,
    min_points_in_bbox: int,
    min_points_used: int,
    depth_percentile: float,
    bbox_candidate_y_min_ratio: float,
    bbox_candidate_y_max_ratio: float,
) -> tuple[dict[str, Any], DetectionPointSelection]:
    """估计单个 2D detection 的 ego 代表点并判断 ROI。"""
    bbox = detection.get("bbox_xyxy")
    if not bbox or len(bbox) != 4:
        return unknown_roi("missing_bbox_xyxy"), empty_selection()

    x1, y1, x2, y2 = [float(value) for value in bbox]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    pixels = projected.pixels
    # 先用 bbox 在图像平面筛出全部 LiDAR 点，再裁掉容易包含地面的下沿区域。
    in_bbox_mask = (
        (pixels[:, 0] >= x1)
        & (pixels[:, 0] <= x2)
        & (pixels[:, 1] >= y1)
        & (pixels[:, 1] <= y2)
    )
    in_bbox_indices = np.flatnonzero(in_bbox_mask)
    points_in_bbox = int(in_bbox_indices.size)
    if points_in_bbox < min_points_in_bbox:
        empty3 = np.empty((0, 3), dtype=np.float64)
        return (
            unknown_roi("insufficient_lidar_points", points_in_bbox=points_in_bbox),
            DetectionPointSelection(
                points_in_bbox,
                0,
                projected.pixels[in_bbox_indices],
                np.empty((0, 2), dtype=np.float64),
                empty3,
                empty3,
            ),
        )

    bbox_height = max(y2 - y1, 1.0)
    candidate_y_min = y1 + bbox_height * bbox_candidate_y_min_ratio
    candidate_y_max = y1 + bbox_height * bbox_candidate_y_max_ratio
    bbox_pixels = projected.pixels[in_bbox_indices]
    candidate_local_mask = (
        (bbox_pixels[:, 1] >= candidate_y_min)
        & (bbox_pixels[:, 1] <= candidate_y_max)
    )
    candidate_indices = in_bbox_indices[candidate_local_mask]
    points_candidate = int(candidate_indices.size)
    if points_candidate < min_points_in_bbox:
        empty3 = np.empty((0, 3), dtype=np.float64)
        return (
            unknown_roi("insufficient_candidate_lidar_points", points_in_bbox=points_in_bbox),
            DetectionPointSelection(
                points_in_bbox,
                0,
                projected.pixels[in_bbox_indices],
                np.empty((0, 2), dtype=np.float64),
                empty3,
                empty3,
            ),
        )

    candidate_camera = projected.points_camera[candidate_indices]
    depths = candidate_camera[:, 2]
    # 取深度较近的一部分点，降低背景点或穿透点对代表点估计的影响。
    order = np.argsort(depths)
    percentile_count = int(math.ceil(points_candidate * depth_percentile))
    use_count = max(min_points_used, percentile_count)
    use_count = min(points_candidate, use_count)
    used_local_indices = order[:use_count]
    used_indices = candidate_indices[used_local_indices]

    used_ego = projected.points_ego[used_indices]
    used_camera = projected.points_camera[used_indices]
    # 用 median 而非 mean，减少离群点影响。
    representative_ego = np.median(used_ego, axis=0)
    representative_camera = np.median(used_camera, axis=0)
    roi = build_roi_info(
        representative_point_ego=representative_ego,
        representative_point_camera=representative_camera,
        points_in_bbox=points_in_bbox,
        points_used=int(use_count),
        roi_config=roi_config,
    )
    return (
        roi,
        DetectionPointSelection(
            points_in_bbox,
            int(use_count),
            projected.pixels[in_bbox_indices],
            projected.pixels[used_indices],
            used_camera,
            used_ego,
        ),
    )
