#!/usr/bin/env python
"""对齐图像检测的命令行入口。

本文件只负责解析 CLI 参数并调用 pipeline，具体检测流程在
`pipelines/aligned_detection.py` 中实现。后续新增模型参数时优先改
`parse_args()`，业务流程尽量不要放回入口文件。
"""

import argparse

from pipelines.aligned_detection import run_aligned_detection


def parse_args() -> argparse.Namespace:
    """解析对齐检测流程所需的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Run aligned GroundingDINO detection from aligned_index.json."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root directory.")
    parser.add_argument("--aligned-index", required=True, help="Path to aligned_index.json.")
    parser.add_argument(
        "--projection-summary",
        default=None,
        help="Optional path to vis_projection_newstruct/projection_summary.json.",
    )
    parser.add_argument(
        "--model-name",
        default="grounding_dino_b",
        choices=["grounding_dino_b"],
        help="Model adapter name. First phase supports GroundingDINO-B only.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output run directory. Defaults to "
            "<dataset-root>/aligned_<model-name>, for example "
            "<dataset-root>/aligned_grounding_dino_b."
        ),
    )
    parser.add_argument(
        "--grounding-config",
        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        help="GroundingDINO config path.",
    )
    parser.add_argument(
        "--grounding-checkpoint",
        default="GroundingDINO/weights/groundingdino_swinb_cogcoor.pth",
        help="GroundingDINO checkpoint path.",
    )
    parser.add_argument(
        "--prompt",
        default="car . pedestrian . cyclist . van . traffic cone .",
        help="GroundingDINO text prompt.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--save-vis", action="store_true", help="Save model-only visualizations.")
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Also process PCD frames whose aligned_index frame valid_time_match is false.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Optional camera subset. Defaults to aligned_index cameras.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max PCD frames to process.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional max unique RGB images to infer. Mainly for quick tests.",
    )
    return parser.parse_args()


def main() -> None:
    """入口函数：把命令行参数交给 pipeline 执行。"""
    args = parse_args()
    run_aligned_detection(args)


if __name__ == "__main__":
    main()
