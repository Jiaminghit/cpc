"""3D 框几何计算与相机投影工具。

任务角色：
    提供 3D 检测框角点生成、齐次坐标变换、相机像素投影、2D bbox 计算等
    几何基础能力，供相机投影、ROI 调试和 BEV 可视化共享。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from utils.calibration import CameraCalibration
from utils.common import round_float


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


# 工作角色：根据 2D 点集计算未裁剪的 xyxy bbox。
# 输入输出：输入 N x 2 的像素点数组，输出 [xmin, ymin, xmax, ymax]。
# 实现思路：分别对 x/y 取最小值和最大值，并用 round_float 统一输出精度。
def bbox_xyxy_from_points(points_2d: np.ndarray) -> list[float]:
    return [
        round_float(np.min(points_2d[:, 0])),
        round_float(np.min(points_2d[:, 1])),
        round_float(np.max(points_2d[:, 0])),
        round_float(np.max(points_2d[:, 1])),
    ]


# 工作角色：将 xyxy bbox 裁剪到图像范围内。
# 输入输出：输入 bbox、图像宽高，输出裁剪后的 [xmin, ymin, xmax, ymax]。
# 实现思路：对四个边界分别用 np.clip 限制在像素合法范围，再统一浮点精度。
def clip_bbox_xyxy(bbox: list[float], width: int, height: int) -> list[float]:
    return [
        round_float(np.clip(bbox[0], 0, width - 1)),
        round_float(np.clip(bbox[1], 0, height - 1)),
        round_float(np.clip(bbox[2], 0, width - 1)),
        round_float(np.clip(bbox[3], 0, height - 1)),
    ]


# 工作角色：由 3D box 参数生成 8 个角点。
# 输入输出：输入中心点 center、尺寸 size_lwh 和绕 z 轴 heading，输出 8 x 3 角点数组。
# 实现思路：先在 box 局部坐标系构造长宽高的一半偏移，再用 heading 构造 z 轴旋转矩阵，最后平移到中心点。
def box_to_corners(center: np.ndarray, size_lwh: np.ndarray, heading: float) -> np.ndarray:
    l, w, h = size_lwh.tolist()
    local = np.asarray(
        [
            [l / 2, w / 2, -h / 2],
            [l / 2, -w / 2, -h / 2],
            [-l / 2, -w / 2, -h / 2],
            [-l / 2, w / 2, -h / 2],
            [l / 2, w / 2, h / 2],
            [l / 2, -w / 2, h / 2],
            [-l / 2, -w / 2, h / 2],
            [-l / 2, w / 2, h / 2],
        ],
        dtype=np.float64,
    )
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    rot_z = np.asarray(
        [
            [cos_h, -sin_h, 0.0],
            [sin_h, cos_h, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return local @ rot_z.T + center


# 工作角色：对一组 3D 点应用 4x4 齐次变换矩阵。
# 输入输出：输入 N x 3 点数组和 4 x 4 transform，输出变换后的 N x 3 点数组。
# 实现思路：给点追加一列 1 形成齐次坐标，右乘变换矩阵转置，再取前三维坐标。
def transform_points(points: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    transformed = homogeneous @ transform_4x4.T
    return transformed[:, :3]


# 工作角色：将相机坐标系下的 3D 点投影到图像像素坐标。
# 输入输出：输入相机坐标点和 CameraCalibration，输出 N x 2 像素点数组。
# 实现思路：根据 distortion_model 选择 OpenCV fisheye 或普通 projectPoints，rvec/tvec 置零表示点已在相机坐标系。
def project_points(points_cam: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    object_points = points_cam.reshape(-1, 1, 3).astype(np.float64)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)

    if calibration.distortion_model.lower() == "fisheye":
        distortion = calibration.distortion_coeffs[:4].reshape(4, 1)
        image_points, _ = cv2.fisheye.projectPoints(
            object_points,
            rvec,
            tvec,
            calibration.camera_matrix,
            distortion,
        )
    else:
        image_points, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            calibration.camera_matrix,
            calibration.distortion_coeffs,
        )
    return image_points.reshape(-1, 2)


# 工作角色：判断投影后的 2D bbox 是否和图像画布有交集。
# 输入输出：输入 N x 2 像素点数组和图像宽高，输出 bool。
# 实现思路：先计算点集 bbox；只要 bbox 完全在图像左/上/右/下之外才判定无交集。
def projected_bbox_intersects(points_2d: np.ndarray, width: int, height: int) -> bool:
    min_x = float(np.min(points_2d[:, 0]))
    max_x = float(np.max(points_2d[:, 0]))
    min_y = float(np.min(points_2d[:, 1]))
    max_y = float(np.max(points_2d[:, 1]))
    return max_x >= 0 and max_y >= 0 and min_x < width and min_y < height
