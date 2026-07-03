from __future__ import annotations

"""点云坐标变换与相机投影工具。"""

from dataclasses import dataclass

import cv2
import numpy as np

from utils.calibration import CameraCalibration


@dataclass(frozen=True)
class ProjectedPointCloud:
    """LiDAR 点投影到图像后的像素坐标及其 camera/ego 坐标。"""

    pixels: np.ndarray
    points_camera: np.ndarray
    points_ego: np.ndarray


def transform_points(points: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    """对 N x 3 点集应用 4x4 齐次变换矩阵。"""
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    transformed = homogeneous @ transform_4x4.T
    return transformed[:, :3]


def project_points(points_cam: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    """把 camera 坐标系下的 3D 点投影为图像像素坐标。"""
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


def project_lidar_to_camera(
    *,
    points_lidar: np.ndarray,
    t_lidar_to_ego: np.ndarray,
    calibration: CameraCalibration,
    image_width: int,
    image_height: int,
    min_depth: float,
) -> ProjectedPointCloud:
    """将 LiDAR 点云转换到 ego/camera，并过滤到图像画布内。"""
    points_ego = transform_points(points_lidar, t_lidar_to_ego)
    points_camera = transform_points(points_ego, calibration.t_ego_to_cam)
    depth_mask = points_camera[:, 2] > min_depth
    points_ego = points_ego[depth_mask]
    points_camera = points_camera[depth_mask]
    if points_camera.size == 0:
        empty2 = np.empty((0, 2), dtype=np.float64)
        empty3 = np.empty((0, 3), dtype=np.float64)
        return ProjectedPointCloud(empty2, empty3, empty3)

    pixels = project_points(points_camera, calibration)
    finite_mask = np.isfinite(pixels).all(axis=1)
    image_mask = (
        finite_mask
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < image_width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < image_height)
    )
    return ProjectedPointCloud(
        pixels=pixels[image_mask],
        points_camera=points_camera[image_mask],
        points_ego=points_ego[image_mask],
    )
