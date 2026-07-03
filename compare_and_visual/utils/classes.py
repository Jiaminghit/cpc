"""类别表和绘图颜色配置。

CLASS_TABLE 会写进 manifest，颜色函数主要服务于 overlay 可视化。
"""

from __future__ import annotations


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


def color_for(class_name: str | None) -> tuple[int, int, int]:
    """按类别返回参考框颜色；未知类别使用默认灰色。"""
    if not class_name:
        return DEFAULT_COLOR
    return CLASS_COLORS.get(class_name, DEFAULT_COLOR)


def contrast_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """给模型框取一个反色，避免和 reference 框颜色完全重叠。"""
    return tuple(255 - int(channel) for channel in color)
