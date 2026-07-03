"""OpenCV 绘图工具。

overlay 和 error visualization 共用这些低层绘图函数，保持标签、虚线框、
3D 角点连线等视觉风格一致。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from utils.bbox import clean_bbox


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


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """绘制带黑色底的标签，保证不同背景下文字可读。"""
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
    """绘制普通 2D 实线框。"""
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
    """用短线段模拟虚线；OpenCV 没有直接的 dashed rectangle API。"""
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
    """绘制模型检测框：虚线边框加可选半透明填充。"""
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
    """根据 8 个 2D 角点绘制 3D bbox 线框。"""
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
