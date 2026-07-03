#!/usr/bin/env python3
"""LiDAR ROI 过滤的命令行入口。

本文件只保留参数解析和入口调用。点云投影、ROI 判断、summary 写出等
主流程集中在 `pipelines/lidar_roi_filter.py`，便于后续测试和复用。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.lidar_roi_filter import run_lidar_roi_filter
from utils.roi import DEFAULT_BOX_SOURCE_FRAME, ROI_AXIS_INDEX


def parse_args() -> argparse.Namespace:
    """解析 LiDAR ROI 过滤流程所需的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Filter aligned GroundingDINO detections using LiDAR-derived ego ROI."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--aligned-index", type=Path, default=None)
    parser.add_argument("--detections-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--box-source-frame", default=DEFAULT_BOX_SOURCE_FRAME)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--min-points-in-bbox", type=int, default=3)
    parser.add_argument("--min-points-used", type=int, default=3)
    parser.add_argument("--depth-percentile", type=float, default=0.2)
    parser.add_argument(
        "--bbox-candidate-y-min-ratio",
        type=float,
        default=0.0,
        help="Top relative y bound inside bbox for LiDAR depth candidates. 0 is bbox top.",
    )
    parser.add_argument(
        "--bbox-candidate-y-max-ratio",
        type=float,
        default=0.85,
        help="Bottom relative y bound inside bbox for LiDAR depth candidates. 1 is bbox bottom.",
    )
    parser.add_argument("--roi-lateral-axis", choices=sorted(ROI_AXIS_INDEX), default="y")
    parser.add_argument("--roi-longitudinal-axis", choices=sorted(ROI_AXIS_INDEX), default="x")
    parser.add_argument("--roi-lateral-min", type=float, default=-50.0)
    parser.add_argument("--roi-lateral-max", type=float, default=50.0)
    parser.add_argument("--roi-longitudinal-min", type=float, default=-50.0)
    parser.add_argument("--roi-longitudinal-max", type=float, default=150.0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Also process records whose frame_valid_time_match is false.",
    )
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--save-debug-vis", action="store_true")
    parser.add_argument("--pcd-cache-size", type=int, default=2)
    parser.add_argument("--projection-cache-size", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    """入口函数：把命令行参数交给 ROI filter pipeline 执行。"""
    args = parse_args()
    run_lidar_roi_filter(args)


if __name__ == "__main__":
    main()
