"""ROI 配置与逐框 ROI 判断工具。

任务角色：
    管理质检 ROI 范围配置，并基于 ego 坐标系中的 3D box 中心点判断目标是否落在 ROI 内。
    当前 ROI 默认表示左右 50m、前 150m、后 50m 的空间范围。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.common import array_to_rounded_list, round_float


ROI_FRAME = "ego"
ROI_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


# 工作角色：描述 ROI 判断与过滤的运行配置。
# 输入输出：作为数据类承载是否启用 ROI、是否过滤 ROI 外目标、使用的坐标轴和范围；输出由 ROI 判断函数读取。
# 实现思路：将 CLI 参数整理成不可变配置对象，避免不同函数之间传递大量零散参数。
@dataclass(frozen=True)
class RoiConfig:
    enabled: bool
    filter_enabled: bool
    frame: str
    lateral_axis: str
    longitudinal_axis: str
    lateral_range_m: tuple[float, float]
    longitudinal_range_m: tuple[float, float]


# 工作角色：将 RoiConfig 转为可写入 JSON 的普通 dict。
# 输入输出：输入 RoiConfig，输出只含基础 Python 类型的 dict。
# 实现思路：逐字段展开配置，并对范围边界使用 round_float 统一精度。
def roi_config_to_dict(roi_config: RoiConfig) -> dict[str, Any]:
    return {
        "enabled": roi_config.enabled,
        "filter_enabled": roi_config.filter_enabled,
        "frame": roi_config.frame,
        "lateral_axis": roi_config.lateral_axis,
        "longitudinal_axis": roi_config.longitudinal_axis,
        "lateral_range_m": [
            round_float(roi_config.lateral_range_m[0]),
            round_float(roi_config.lateral_range_m[1]),
        ],
        "longitudinal_range_m": [
            round_float(roi_config.longitudinal_range_m[0]),
            round_float(roi_config.longitudinal_range_m[1]),
        ],
    }


# 工作角色：生成 ROI 相关计数器的初始值。
# 输入输出：无输入，输出包含各类 ROI 统计字段且值为 0 的 dict。
# 实现思路：集中定义统计字段，保证缺图、跳过和正常投影分支写出的 summary 结构一致。
def empty_roi_counts() -> dict[str, int]:
    return {
        "boxes_in_roi": 0,
        "boxes_outside_roi": 0,
        "boxes_projected_in_roi": 0,
        "boxes_projected_outside_roi": 0,
        "boxes_skipped_outside_roi": 0,
    }


# 工作角色：计算单个 3D box 中心点的 ROI 状态与位置明细。
# 输入输出：输入 ego 坐标下的中心点和 RoiConfig，输出包含 in_roi、center_ego、横纵向坐标和范围的 dict。
# 实现思路：根据配置选择 lateral/longitudinal 对应的 ego 轴，然后判断中心点是否落在两个范围闭区间内。
def build_roi_info(center_ego: np.ndarray, roi_config: RoiConfig) -> dict[str, Any]:
    if not roi_config.enabled:
        return {
            "enabled": False,
            "in_roi": None,
            "frame": roi_config.frame,
            "center_ego": array_to_rounded_list(center_ego),
        }

    lateral_idx = ROI_AXIS_INDEX[roi_config.lateral_axis]
    longitudinal_idx = ROI_AXIS_INDEX[roi_config.longitudinal_axis]
    lateral = float(center_ego[lateral_idx])
    longitudinal = float(center_ego[longitudinal_idx])
    lateral_min, lateral_max = roi_config.lateral_range_m
    longitudinal_min, longitudinal_max = roi_config.longitudinal_range_m
    in_roi = (
        lateral_min <= lateral <= lateral_max
        and longitudinal_min <= longitudinal <= longitudinal_max
    )

    return {
        "enabled": True,
        "in_roi": in_roi,
        "frame": roi_config.frame,
        "center_ego": array_to_rounded_list(center_ego),
        "lateral_axis": roi_config.lateral_axis,
        "longitudinal_axis": roi_config.longitudinal_axis,
        "lateral_m": round_float(lateral),
        "longitudinal_m": round_float(longitudinal),
        "lateral_range_m": [
            round_float(lateral_min),
            round_float(lateral_max),
        ],
        "longitudinal_range_m": [
            round_float(longitudinal_min),
            round_float(longitudinal_max),
        ],
    }
