from __future__ import annotations

"""ROI 过滤 summary 统计工具。"""

from typing import Any

import numpy as np

from utils.io import round_float


def update_counter_block(block: dict[str, Any], roi: dict[str, Any], class_name: str | None = None) -> None:
    """更新总计/按相机/按类别的 ROI 计数块。"""
    block["detections_total"] = block.get("detections_total", 0) + 1
    if roi.get("in_roi") is True:
        block["detections_in_roi"] = block.get("detections_in_roi", 0) + 1
    elif roi.get("in_roi") is False:
        block["detections_outside_roi"] = block.get("detections_outside_roi", 0) + 1
    else:
        block["detections_unknown_roi"] = block.get("detections_unknown_roi", 0) + 1
        reason = roi.get("reason") or "unknown"
        reasons = block.setdefault("unknown_reasons", {})
        reasons[reason] = reasons.get(reason, 0) + 1
    if class_name is not None:
        classes = block.setdefault("by_class", {})
        class_block = classes.setdefault(class_name, {})
        update_counter_block(class_block, roi, None)


def ratio(numerator: int, denominator: int) -> float | None:
    """计算比例；分母为 0 时返回 None，避免写出无意义数值。"""
    if denominator == 0:
        return None
    return round_float(numerator / denominator)


def summarize_points(values: list[int]) -> dict[str, Any]:
    """汇总 bbox 内点数或实际使用点数的 min/median/max。"""
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": int(np.min(arr)),
        "median": round_float(np.median(arr)),
        "max": int(np.max(arr)),
    }
