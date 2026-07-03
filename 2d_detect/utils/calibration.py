from __future__ import annotations

"""标定读取与传感器坐标系转换工具。"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """相机投影所需的内参、畸变、ego->camera 外参和图像尺寸。"""

    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    t_ego_to_cam: np.ndarray
    width: int
    height: int
    distortion_model: str


def sensor_name(sensor: dict[str, Any]) -> str:
    """兼容 `sensor_name` 和 `name` 两种标定字段。"""
    return str(sensor.get("sensor_name") or sensor.get("name") or "<unknown>")


def find_sensor(calib_json: dict[str, Any], name: str) -> dict[str, Any]:
    """从 calibration JSON 中查找指定传感器。"""
    for sensor in calib_json.get("sensors", []):
        if sensor.get("sensor_name") == name or sensor.get("name") == name:
            return sensor
    raise KeyError(f"sensor not found in calibration: {name}")


def sensor_transform_matrix(sensor: dict[str, Any]) -> np.ndarray:
    """读取 sensor 外参矩阵，返回 4x4 numpy 矩阵。"""
    extrinsics = sensor.get("extrinsics") or sensor.get("extrinsic") or {}
    if "transform_matrix_4x4" in extrinsics:
        return np.asarray(extrinsics["transform_matrix_4x4"], dtype=np.float64)

    transform_matrix = extrinsics.get("transform_matrix")
    if transform_matrix is None:
        raise KeyError(f"sensor has no transform_matrix: {sensor_name(sensor)}")

    matrix = np.asarray(transform_matrix, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f"transform_matrix must have 16 values for {sensor_name(sensor)}")
    return matrix.reshape(4, 4)


def intrinsic_camera_matrix(intrinsic: dict[str, Any]) -> np.ndarray:
    """构造 OpenCV 投影使用的 3x3 相机内参矩阵。"""
    if "camera_matrix" in intrinsic:
        return np.asarray(intrinsic["camera_matrix"], dtype=np.float64)
    return np.asarray(
        [
            [float(intrinsic["fx"]), 0.0, float(intrinsic["cx"])],
            [0.0, float(intrinsic["fy"]), float(intrinsic["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def load_source_to_ego(calib_json: dict[str, Any], source_frame: str) -> np.ndarray:
    """读取点云源坐标系到 ego 坐标系的变换矩阵。"""
    if source_frame == "ego":
        return np.eye(4, dtype=np.float64)
    sensor = find_sensor(calib_json, source_frame)
    return sensor_transform_matrix(sensor)


def load_camera_calibration(calib_json: dict[str, Any], camera_name: str) -> CameraCalibration:
    """读取指定相机的投影标定，并预先计算 ego->camera 外参。"""
    sensor = find_sensor(calib_json, camera_name)
    intrinsics = sensor.get("intrinsics") or []
    if not intrinsics:
        raise ValueError(f"camera has no intrinsics: {camera_name}")

    intrinsic = intrinsics[0]
    camera_matrix = intrinsic_camera_matrix(intrinsic)
    distortion_coeffs = np.asarray(intrinsic.get("distortion_coeffs", []), dtype=np.float64)
    t_cam_to_ego = sensor_transform_matrix(sensor)
    return CameraCalibration(
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        t_ego_to_cam=np.linalg.inv(t_cam_to_ego),
        width=int(intrinsic["width"]),
        height=int(intrinsic["height"]),
        distortion_model=str(intrinsic.get("distortion_model", "")),
    )
