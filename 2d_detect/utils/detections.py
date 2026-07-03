from __future__ import annotations

"""检测结果格式辅助工具。"""

from typing import Any


def clone_detection_for_alignment(
    detection: dict[str, Any],
    pcd_timestamp: int,
    camera: str,
    idx: int,
    model_name: str,
) -> dict[str, Any]:
    """复制图像级 detection，并替换为 PCD-camera 级别唯一 det_id。"""
    cloned = dict(detection)
    cloned["det_id"] = f"{model_name}:{pcd_timestamp}:{camera}:{idx}"
    return cloned
