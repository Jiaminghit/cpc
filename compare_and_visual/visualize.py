#!/usr/bin/env python3
"""叠加可视化 3D 投影参考框和模型检测框。

这是 projection/model comparison workflow 的 Stage 3A：只生成可视化图片和
summary 元数据，不计算 IoU、TP/FP/FN、precision 或 recall。
"""

from __future__ import annotations

import argparse

from pipelines.projection_model_overlay import run_projection_model_overlay
from utils.drawing import DEFAULT_MODEL_MASK_ALPHA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay projection reference boxes and model detections."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root directory.")
    parser.add_argument("--projection-jsonl", required=True, help="projections_aligned.jsonl path.")
    parser.add_argument("--model-jsonl", required=True, help="detections_aligned.jsonl path.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <dataset-root>/compare_grounding_dino_b_projection.",
    )
    parser.add_argument("--save-vis", action="store_true", help="Write overlay images.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional max aligned records.")
    parser.add_argument("--cameras", nargs="+", default=None, help="Optional camera subset.")
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Also visualize records whose frame_valid_time_match is false.",
    )
    parser.add_argument(
        "--draw-projection-corners",
        action="store_true",
        help="Compatibility flag. Reference boxes are now always drawn from corners_2d.",
    )
    parser.add_argument(
        "--model-name",
        default="grounding_dino_b",
        help="Model name recorded in manifest/summary.",
    )
    parser.add_argument(
        "--model-mask-alpha",
        type=float,
        default=DEFAULT_MODEL_MASK_ALPHA,
        help="Transparent fill alpha for model detection boxes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 可视化入口保持轻量，具体读写、对齐、绘图逻辑交给 pipeline 层。
    run_projection_model_overlay(args)


if __name__ == "__main__":
    main()
