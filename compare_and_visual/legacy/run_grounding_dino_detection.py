#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import torch
from torchvision.ops import box_convert

from groundingdino.util.inference import annotate, load_image, load_model, predict


DEFAULT_CLASSES = ["Car", "Pedestrian", "Cyclist", "Van", "Traffic_cone"]
DEFAULT_ALIASES = {
    "Car": ["car", "vehicle"],
    "Pedestrian": ["pedestrian", "person"],
    "Cyclist": ["person riding bicycle", "person riding bike", "bicyclist", "cyclist"],
    "Van": ["minivan", "van"],
    "Traffic_cone": ["traffic cone", "traffic_cone", "cone"],
}
CLASS_PRIORITY = {
    "Traffic_cone": 0,
    "Cyclist": 1,
    "Van": 2,
    "Pedestrian": 3,
    "Car": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GroundingDINO on one image or a directory and save detections as JSONL."
    )
    parser.add_argument("--input", required=True, help="Input image file or directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSONL and optional visualizations.")
    parser.add_argument(
        "--config",
        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        help="GroundingDINO config path.",
    )
    parser.add_argument(
        "--checkpoint",
        default="GroundingDINO/weights/groundingdino_swinb_cogcoor.pth",
        help="GroundingDINO checkpoint path.",
    )
    parser.add_argument(
        "--prompt",
        default="car . pedestrian . cyclist . van . traffic cone .",
        help="Text prompt used by GroundingDINO.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directory.")
    parser.add_argument("--save-vis", action="store_true", help="Save annotated images.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of images to process.")
    parser.add_argument(
        "--jsonl-name",
        default="detections.jsonl",
        help="Output JSONL filename inside --output-dir.",
    )
    return parser.parse_args()


def iter_images(input_path: Path, recursive: bool) -> Iterable[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    if input_path.is_file():
        if input_path.suffix.lower() in suffixes:
            yield input_path
        return

    pattern = "**/*" if recursive else "*"
    for path in sorted(input_path.glob(pattern)):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def build_alias_index(aliases: Dict[str, List[str]]) -> List[tuple]:
    items = []
    for class_name, class_aliases in aliases.items():
        for alias in class_aliases:
            items.append((alias.lower(), class_name))
    return sorted(items, key=lambda item: (-len(item[0]), CLASS_PRIORITY[item[1]]))


def canonicalize_label(raw_phrase: str, alias_index: List[tuple]) -> Optional[str]:
    normalized = raw_phrase.lower().replace("_", " ").strip()
    for alias, class_name in alias_index:
        if alias in normalized:
            return class_name
    return None


def tensor_to_list(value: torch.Tensor) -> List[float]:
    return [round(float(v), 6) for v in value.tolist()]


def make_record(
    image_path: Path,
    input_root: Path,
    width: int,
    height: int,
    boxes_cxcywh_norm: torch.Tensor,
    scores: torch.Tensor,
    phrases: List[str],
    alias_index: List[tuple],
    prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> Dict:
    boxes_xyxy = box_convert(
        boxes=boxes_cxcywh_norm * torch.tensor([width, height, width, height]),
        in_fmt="cxcywh",
        out_fmt="xyxy",
    )

    try:
        relative_path = str(image_path.relative_to(input_root))
    except ValueError:
        relative_path = image_path.name

    detections = []
    for box_xyxy, box_norm, score, phrase in zip(boxes_xyxy, boxes_cxcywh_norm, scores, phrases):
        detections.append(
            {
                "class_name": canonicalize_label(phrase, alias_index),
                "label_raw": phrase,
                "score": round(float(score), 6),
                "bbox_xyxy": tensor_to_list(box_xyxy),
                "bbox_cxcywh_norm": tensor_to_list(box_norm),
            }
        )

    return {
        "image_path": str(image_path),
        "relative_path": relative_path,
        "image_id": image_path.stem,
        "camera": image_path.parent.name,
        "width": width,
        "height": height,
        "model": "GroundingDINO-B",
        "checkpoint": "groundingdino_swinb_cogcoor.pth",
        "prompt": prompt,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "detections": detections,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    input_root = input_path if input_path.is_dir() else input_path.parent
    image_paths = list(iter_images(input_path, args.recursive))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {input_path}")

    alias_index = build_alias_index(DEFAULT_ALIASES)
    model = load_model(args.config, args.checkpoint, device=args.device)
    jsonl_path = output_dir / args.jsonl_name

    total_detections = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        for idx, image_path in enumerate(image_paths, start=1):
            image_source, image_tensor = load_image(str(image_path))
            boxes, scores, phrases = predict(
                model=model,
                image=image_tensor,
                caption=args.prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=args.device,
            )

            height, width = image_source.shape[:2]
            record = make_record(
                image_path=image_path,
                input_root=input_root,
                width=width,
                height=height,
                boxes_cxcywh_norm=boxes,
                scores=scores,
                phrases=phrases,
                alias_index=alias_index,
                prompt=args.prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_detections += len(record["detections"])

            if args.save_vis:
                annotated = annotate(image_source, boxes, scores, phrases)
                vis_path = vis_dir / f"{image_path.stem}.jpg"
                cv2.imwrite(str(vis_path), annotated)

            print(f"[{idx}/{len(image_paths)}] {image_path} detections={len(record['detections'])}")

    summary = {
        "input": str(input_path),
        "output_jsonl": str(jsonl_path),
        "num_images": len(image_paths),
        "num_detections": total_detections,
        "classes": DEFAULT_CLASSES,
        "prompt": args.prompt,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Done. Wrote {jsonl_path}")


if __name__ == "__main__":
    main()
