from __future__ import annotations

"""检测模型适配器工厂。

pipeline 只依赖这里返回的统一 adapter，不直接关心底层是 GroundingDINO、
YOLO 还是其他模型。后续接入新模型时，在这里新增分支即可。
"""

from argparse import Namespace
from typing import Any

from models.grounding_dino import GroundingDinoAdapter


def build_model_adapter(args: Namespace, *, resolve_project_path: Any) -> GroundingDinoAdapter:
    """根据 `--model-name` 创建对应的检测模型 adapter。"""
    if args.model_name == "grounding_dino_b":
        return GroundingDinoAdapter(
            config_path=resolve_project_path(args.grounding_config),
            checkpoint_path=resolve_project_path(args.grounding_checkpoint),
            prompt=args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
        )

    raise ValueError(f"Unsupported model adapter: {args.model_name}")
