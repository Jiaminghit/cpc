from __future__ import annotations

"""GroundingDINO 模型适配层。

职责：
- 处理 GroundingDINO 代码路径、模型加载和推理调用。
- 将 GroundingDINO 的 box/score/phrase 输出转换成项目统一检测格式。
- 为 pipeline 提供 `predict()` 与 `render_detections()` 两个稳定接口。
"""

import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUNDING_DINO_ROOT = PROJECT_ROOT / "GroundingDINO"
if str(GROUNDING_DINO_ROOT) not in sys.path:
    sys.path.insert(0, str(GROUNDING_DINO_ROOT))

import cv2
import torch
from torchvision.ops import box_convert

from groundingdino.util.inference import annotate, load_image, load_model, predict

from models.classes import build_alias_index, canonicalize_label


def tensor_to_list(value: torch.Tensor) -> list[float]:
    """将 tensor 转为可 JSON 序列化的浮点列表，并统一保留 6 位小数。"""
    return [round(float(v), 6) for v in value.tolist()]


def make_detections(
    image_width: int,
    image_height: int,
    boxes_cxcywh_norm: torch.Tensor,
    scores: torch.Tensor,
    phrases: list[str],
    alias_index: list[tuple[str, str]],
    model_name: str,
    image_key: str,
) -> list[dict[str, Any]]:
    """把 GroundingDINO 原始输出转换为统一的 detection dict 列表。"""
    # GroundingDINO 输出为归一化 cxcywh，这里转为像素 xyxy 供后续 ROI 和评估使用。
    boxes_xyxy = box_convert(
        boxes=boxes_cxcywh_norm * torch.tensor([image_width, image_height, image_width, image_height]),
        in_fmt="cxcywh",
        out_fmt="xyxy",
    )

    detections = []
    for idx, (box_xyxy, box_norm, score, phrase) in enumerate(zip(boxes_xyxy, boxes_cxcywh_norm, scores, phrases)):
        class_name, class_id = canonicalize_label(phrase, alias_index)
        detections.append(
            {
                "det_id": f"{model_name}:{image_key}:{idx}",
                "class_name": class_name,
                "class_id": class_id,
                "label_raw": phrase,
                "score": round(float(score), 6),
                "bbox_xyxy": tensor_to_list(box_xyxy),
                "bbox_cxcywh_norm": tensor_to_list(box_norm),
                "segmentation": None,
                "extra": {},
            }
        )
    return detections


class GroundingDinoAdapter:
    """GroundingDINO-B 的统一模型 adapter。"""

    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        device: str,
    ):
        """加载模型权重，并记录后续 manifest/summary 需要的模型元信息。"""
        self.model_name = "grounding_dino_b"
        self.model_family = "open_vocab_detection"
        self.config_path = str(config_path)
        self.checkpoint_path = str(checkpoint_path)
        self.checkpoint_name = checkpoint_path.name
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.alias_index = build_alias_index()
        self.model = load_model(str(config_path), str(checkpoint_path), device=device)

    def predict(self, image_path: Path, image_key: str) -> dict[str, Any]:
        """对单张 RGB 图像执行检测，返回图像尺寸、检测结果和可视化图。"""
        image_source, image_tensor = load_image(str(image_path))
        boxes, scores, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=self.prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        height, width = image_source.shape[:2]
        detections = make_detections(
            image_width=width,
            image_height=height,
            boxes_cxcywh_norm=boxes,
            scores=scores,
            phrases=phrases,
            alias_index=self.alias_index,
            model_name=self.model_name,
            image_key=image_key,
        )
        if detections:
            vis = annotate(image_source, boxes, scores, phrases)
        else:
            vis = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
        return {
            "image_width": width,
            "image_height": height,
            "detections": detections,
            "vis": vis,
        }

    def render_detections(self, image_path: Path, detections: list[dict[str, Any]]) -> Any:
        """根据已保存的统一检测结果重新渲染可视化图。"""
        image_source, _ = load_image(str(image_path))
        if not detections:
            return cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)

        boxes = torch.tensor([det["bbox_cxcywh_norm"] for det in detections], dtype=torch.float32)
        scores = torch.tensor([det["score"] for det in detections], dtype=torch.float32)
        phrases = [det["label_raw"] for det in detections]
        return annotate(image_source, boxes, scores, phrases)

    def model_block(self) -> dict[str, Any]:
        """返回写入 manifest/summary 的模型描述。"""
        return {
            "name": self.model_name,
            "family": self.model_family,
            "checkpoint": self.checkpoint_name,
            "config": Path(self.config_path).name,
        }

    def params_block(self) -> dict[str, Any]:
        """返回写入 manifest/summary 的推理参数。"""
        return {
            "prompt": self.prompt,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
        }
