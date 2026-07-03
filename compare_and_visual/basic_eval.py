#!/usr/bin/env python3
"""评估 ROI 内 3D 投影框和 GroundingDINO 检测框的 2D 一致性。

这个文件只作为薄 CLI 入口：负责解析命令行参数，真正的评估主流程放在
pipelines/projection_model_eval.py 中，便于后续复用和测试。
"""

from __future__ import annotations

import argparse

from pipelines.projection_model_eval import run_projection_model_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ROI 2D consistency between projected 3D boxes and GroundingDINO ROI detections."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root directory.")
    parser.add_argument("--projection-jsonl", required=True, help="ROI projections_aligned.jsonl path.")
    parser.add_argument("--dino-jsonl", required=True, help="ROI detections_aligned_roi_only.jsonl path.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <dataset-root>/compare_grounding_dino_b_roi_projection_qc.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--low-iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument(
        "--class-aware-matching",
        action="store_true",
        help="Only allow candidate pairs with the same class_name.",
    )
    parser.add_argument(
        "--allow-class-mismatch",
        action="store_true",
        default=True,
        help="Compatibility flag; class mismatch is allowed by default unless --class-aware-matching is set.",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Also evaluate records whose frame_valid_time_match is false.",
    )
    parser.add_argument(
        "--save-error-vis",
        action="store_true",
        help="Save lightweight visualizations for non-TP records.",
    )
    parser.add_argument(
        "--input-scope",
        choices=["roi_filtered", "all"],
        default="roi_filtered",
        help=(
            "Whether input JSONL files are expected to contain ROI-filtered boxes. "
            "Use 'all' for unfiltered projection/detection JSONL files."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # CLI 层不做业务处理，直接把参数交给 pipeline 层。
    run_projection_model_eval(args)


if __name__ == "__main__":
    main()
