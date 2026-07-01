"""OpenCV 可视化绘制工具。

任务角色：
    封装相机图像投影结果的基础绘制逻辑，目前主要负责绘制 3D box 的 12 条边和文本标签。
"""

from __future__ import annotations

import cv2
import numpy as np

from utils.geometry import BOX_EDGES


# 工作角色：在图像上绘制一个投影后的 3D box 线框和标签。
# 输入输出：输入 OpenCV 图像、8 个角点的像素坐标、标签文本和 BGR 颜色；直接原地修改 image，无返回值。
# 实现思路：用 BOX_EDGES 定义的边连接角点，线段先通过 cv2.clipLine 裁剪到画布内，再绘制标签文本。
def draw_box_edges(
    image: np.ndarray,
    points_2d: np.ndarray,
    label: str,
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    rect = (0, 0, width, height)
    int_points = np.rint(points_2d).astype(np.int32)

    for start, end in BOX_EDGES:
        p1 = tuple(int_points[start].tolist())
        p2 = tuple(int_points[end].tolist())
        ok, clipped_p1, clipped_p2 = cv2.clipLine(rect, p1, p2)
        if ok:
            cv2.line(image, clipped_p1, clipped_p2, color, 2, cv2.LINE_AA)

    label_x = int(np.clip(np.min(points_2d[:, 0]), 0, width - 1))
    label_y = int(np.clip(np.min(points_2d[:, 1]) - 6, 16, height - 1))
    cv2.putText(
        image,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
