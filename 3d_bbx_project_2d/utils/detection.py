"""3D 检测框数据结构与 prelabel-model 读取工具。

任务角色：
    将数据集 `pcd/prelabel-model/*.json` 中的 GeoJSON 风格检测结果
    解析为统一的 `DetectionBox` 对象，供相机投影和 BEV 可视化复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.common import load_json


# 工作角色：统一表示一个 3D 目标检测框。
# 输入输出：作为数据类承载 object_id、类别、置信度、中心点、长宽高和 heading；输出由调用方直接读取字段。
# 实现思路：使用 frozen dataclass 保持对象不可变，避免投影过程中意外修改原始检测数据。
@dataclass(frozen=True)
class DetectionBox:
    object_id: str
    category: str
    score: float
    center: np.ndarray
    size_lwh: np.ndarray
    heading: float


# 工作角色：从 prelabel-model JSON 文件中加载并过滤 3D 检测框。
# 输入输出：输入检测 JSON 路径和 score_threshold，输出 DetectionBox 列表。
# 实现思路：遍历 features，读取 coordinates 中的 center/size/heading 和 properties 中的 id/type/score，
#          跳过坐标缺失或置信度低于阈值的目标，并将数值字段转为 numpy 数组。
def load_detection_boxes(det_json_path: Path, score_threshold: float) -> list[DetectionBox]:
    det_json = load_json(det_json_path)
    boxes: list[DetectionBox] = []
    for feature in det_json.get("features", []):
        coordinates = feature.get("geometry", {}).get("coordinates", [])
        properties = feature.get("properties", {})
        if len(coordinates) < 3:
            continue

        score = float(properties.get("score", 0.0))
        if score < score_threshold:
            continue

        center = np.asarray(coordinates[0], dtype=np.float64)
        size_lwh = np.asarray(coordinates[1], dtype=np.float64)
        heading = float(coordinates[2][2])
        boxes.append(
            DetectionBox(
                object_id=str(properties.get("id", "")),
                category=str(properties.get("type", "unknown")).lower(),
                score=score,
                center=center,
                size_lwh=size_lwh,
                heading=heading,
            )
        )
    return boxes
