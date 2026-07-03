from __future__ import annotations

"""OpenCV 可视化绘制工具。

普通可视化用于查看 ROI 过滤后的目标，debug 可视化额外显示 bbox 内
LiDAR 点和实际用于估计代表点的点。
"""

from typing import Any

import cv2
import numpy as np


ROI_IN_COLOR = (60, 220, 60)
ROI_OUT_COLOR = (40, 40, 255)
ROI_UNKNOWN_COLOR = (170, 170, 170)
POINT_IN_BBOX_COLOR = (255, 180, 40)
POINT_USED_COLOR = (40, 220, 255)
CLASS_COLORS = {
    "Car": (255, 80, 40),
    "Van": (255, 220, 40),
    "Pedestrian": (60, 220, 60),
    "Cyclist": (40, 220, 255),
    "Traffic_cone": (40, 40, 255),
}
DEFAULT_CLASS_COLOR = (220, 220, 220)


def color_for_roi(roi: dict[str, Any]) -> tuple[int, int, int]:
    """根据 ROI 状态选择框颜色。"""
    if roi.get("in_roi") is True:
        return ROI_IN_COLOR
    if roi.get("in_roi") is False:
        return ROI_OUT_COLOR
    return ROI_UNKNOWN_COLOR


def color_for_class(detection: dict[str, Any]) -> tuple[int, int, int]:
    """根据类别选择框颜色，未知类别使用默认灰色。"""
    class_name = detection.get("class_name")
    return CLASS_COLORS.get(class_name, DEFAULT_CLASS_COLOR)


def format_roi_position(roi: dict[str, Any]) -> str:
    """格式化 ROI 横纵向位置，供调试可视化文本显示。"""
    longitudinal = roi.get("longitudinal_m")
    lateral = roi.get("lateral_m")
    if isinstance(longitudinal, (int, float)) and isinstance(lateral, (int, float)):
        return f"long={float(longitudinal):.2f}m lat={float(lateral):.2f}m"
    return "long=na lat=na"


def draw_detection(
    image: np.ndarray,
    detection: dict[str, Any],
    *,
    draw_unknown: bool = True,
    color_mode: str = "roi",
    include_roi_status: bool = True,
) -> None:
    """在图像上绘制 detection bbox 和标签。"""
    roi = detection.get("lidar_roi", {})
    if roi.get("in_roi") is None and not draw_unknown:
        return
    bbox = detection.get("bbox_xyxy")
    if not bbox or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    height, width = image.shape[:2]
    x1 = int(np.clip(x1, 0, width - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    y2 = int(np.clip(y2, 0, height - 1))
    color = color_for_class(detection) if color_mode == "class" else color_for_roi(roi)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    class_name = detection.get("class_name") or detection.get("label_raw") or "unknown"
    score = detection.get("score")
    if include_roi_status:
        status = "roi" if roi.get("in_roi") is True else "out" if roi.get("in_roi") is False else "unknown"
        position = format_roi_position(roi)
        label = (
            f"{class_name} {score:.2f} {status} {position}"
            if isinstance(score, (int, float))
            else f"{class_name} {status} {position}"
        )
    else:
        label = f"{class_name} {score:.2f}" if isinstance(score, (int, float)) else str(class_name)
    label_y = int(np.clip(y1 - 6, 16, height - 1))
    cv2.putText(
        image,
        label,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_points(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
    """绘制投影后的 LiDAR 像素点；点过多时做简单下采样。"""
    if points.size == 0:
        return
    height, width = image.shape[:2]
    rounded = np.rint(points).astype(np.int32)
    max_points = min(2000, rounded.shape[0])
    if rounded.shape[0] > max_points:
        step = max(1, rounded.shape[0] // max_points)
        rounded = rounded[::step]
    for x, y in rounded:
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
