"""标定读取与坐标系变换工具。

任务角色：
    负责从 struct_json calibration 中查找传感器、读取内外参、构造相机投影所需参数，
    并提供 3D box 源坐标系到 ego 坐标系的变换矩阵。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils.common import load_json


# 工作角色：统一表示一个相机的投影标定参数。
# 输入输出：作为数据类承载相机内参、畸变参数、ego->camera 外参、图像尺寸和畸变模型。
# 实现思路：`load_calibration()` 从原始标定 JSON 中解析字段后创建该对象，后续投影函数只依赖这个稳定结构。
@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    t_ego_to_cam: np.ndarray
    width: int
    height: int
    distortion_model: str


# 工作角色：在标定 JSON 中查找指定传感器配置。
# 输入输出：输入完整 calibration dict 和 sensor_name，输出对应 sensor dict。
# 实现思路：兼容 `sensor_name` 和 `name` 两种字段命名；找不到时抛出 KeyError，尽早暴露配置问题。
def find_sensor(calib_json: dict[str, Any], sensor_name: str) -> dict[str, Any]:
    for sensor in calib_json.get("sensors", []):
        if sensor.get("sensor_name") == sensor_name or sensor.get("name") == sensor_name:
            return sensor
    raise KeyError(f"sensor not found in calibration: {sensor_name}")


# 工作角色：从 sensor 配置中解析 4x4 外参矩阵。
# 输入输出：输入 sensor dict，输出 numpy.ndarray 形状为 (4, 4)。
# 实现思路：优先读取 `transform_matrix_4x4`；否则读取长度为 16 的 `transform_matrix` 并 reshape。
def sensor_transform_matrix(sensor: dict[str, Any]) -> np.ndarray:
    extrinsics = sensor.get("extrinsics") or sensor.get("extrinsic") or {}
    if "transform_matrix_4x4" in extrinsics:
        return np.asarray(extrinsics["transform_matrix_4x4"], dtype=np.float64)

    transform_matrix = extrinsics.get("transform_matrix")
    if transform_matrix is None:
        sensor_name = sensor.get("sensor_name") or sensor.get("name") or "<unknown>"
        raise KeyError(f"sensor has no transform_matrix: {sensor_name}")

    matrix = np.asarray(transform_matrix, dtype=np.float64)
    if matrix.size != 16:
        sensor_name = sensor.get("sensor_name") or sensor.get("name") or "<unknown>"
        raise ValueError(f"transform_matrix must have 16 values for {sensor_name}")
    return matrix.reshape(4, 4)


# 工作角色：构造相机内参矩阵 K。
# 输入输出：输入单个 intrinsic dict，输出 3x3 numpy 相机矩阵。
# 实现思路：如果标定中已有 `camera_matrix` 就直接使用；否则用 fx/fy/cx/cy 拼成标准针孔模型内参矩阵。
def intrinsic_camera_matrix(intrinsic: dict[str, Any]) -> np.ndarray:
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


# 工作角色：加载 3D box 源坐标系到 ego 坐标系的变换。
# 输入输出：输入标定文件路径和 box_source_frame 名称，输出 4x4 变换矩阵 T_box_source_to_ego。
# 实现思路：当源坐标系已经是 ego 时返回单位矩阵；否则在标定中查找同名传感器并读取其外参矩阵。
def load_box_source_transform(calib_path: Path, box_source_frame: str) -> np.ndarray:
    if box_source_frame == "ego":
        return np.eye(4, dtype=np.float64)

    calib_json = load_json(calib_path)
    sensor = find_sensor(calib_json, box_source_frame)
    return sensor_transform_matrix(sensor)


# 工作角色：加载指定相机的完整投影标定。
# 输入输出：输入标定文件路径和 camera_name，输出 CameraCalibration。
# 实现思路：读取相机内参和畸变参数，读取 camera->ego 外参后求逆得到 ego->camera，供 3D 点投影到像素使用。
def load_calibration(calib_path: Path, camera_name: str) -> CameraCalibration:
    calib_json = load_json(calib_path)
    sensor = find_sensor(calib_json, camera_name)
    intrinsics = sensor.get("intrinsics") or []
    if not intrinsics:
        raise ValueError(f"camera has no intrinsics: {camera_name}")

    intrinsic = intrinsics[0]
    camera_matrix = intrinsic_camera_matrix(intrinsic)
    distortion_coeffs = np.asarray(intrinsic.get("distortion_coeffs", []), dtype=np.float64)
    t_cam_to_ego = sensor_transform_matrix(sensor)
    t_ego_to_cam = np.linalg.inv(t_cam_to_ego)

    return CameraCalibration(
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        t_ego_to_cam=t_ego_to_cam,
        width=int(intrinsic["width"]),
        height=int(intrinsic["height"]),
        distortion_model=str(intrinsic.get("distortion_model", "")),
    )
