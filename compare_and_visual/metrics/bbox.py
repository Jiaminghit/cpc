"""bbox 几何工具。

这里统一处理 bbox 清洗、面积、IoU、中心点距离等基础几何量。上层 matching
和 record 评估都依赖这些函数，避免在不同模块里重复实现 bbox 逻辑。
"""

from __future__ import annotations

import math
from typing import Any


def finite_float(value: Any) -> float | None:
    """把输入转成有限浮点数；非法、NaN、inf 都按 None 处理。"""
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
    """将任意 xyxy bbox 整理成图像范围内的有效浮点框。

    输入可能出现 x1/x2 或 y1/y2 反向、越界、空框等情况。这里统一修正反向
    和裁剪越界；如果裁剪后面积为 0，则返回 None，让上层记录 warning 并跳过。
    """
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
    """计算两个 xyxy bbox 的 IoU；没有交集或 union 异常时返回 0。"""
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
    """投影框面积 / DINO 框面积，用来辅助判断框大小偏差。"""
    dino_area = bbox_area(dino_box)
    if dino_area <= 0.0:
        return None
    return bbox_area(projection_box) / dino_area
