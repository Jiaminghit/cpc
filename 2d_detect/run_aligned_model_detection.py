#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUNDING_DINO_ROOT = PROJECT_ROOT / "GroundingDINO"
if str(GROUNDING_DINO_ROOT) not in sys.path:
    sys.path.insert(0, str(GROUNDING_DINO_ROOT))

import cv2
import torch
from torchvision.ops import box_convert

from groundingdino.util.inference import annotate, load_image, load_model, predict


CLASS_TABLE = [
    {"class_id": 0, "class_name": "Car"},
    {"class_id": 1, "class_name": "Pedestrian"},
    {"class_id": 2, "class_name": "Cyclist"},
    {"class_id": 3, "class_name": "Van"},
    {"class_id": 4, "class_name": "Traffic_cone"},
]
CLASS_ID_BY_NAME = {item["class_name"]: item["class_id"] for item in CLASS_TABLE}
ALIASES = {
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
        "--valid-time-only",
        action="store_true",
        help="Only process camera images whose aligned_index valid_time_match is true.",
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


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_dataset_path(dataset_root: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == dataset_root.name:
        return dataset_root.parent / path
    return dataset_root / path


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def build_alias_index() -> List[Tuple[str, str]]:
    items = []
    for class_name, aliases in ALIASES.items():
        for alias in aliases:
            items.append((alias.lower(), class_name))
    return sorted(items, key=lambda item: (-len(item[0]), CLASS_PRIORITY[item[1]]))


def canonicalize_label(raw_phrase: str, alias_index: List[Tuple[str, str]]) -> Tuple[Optional[str], Optional[int]]:
    normalized = raw_phrase.lower().replace("_", " ").strip()
    for alias, class_name in alias_index:
        if alias in normalized:
            return class_name, CLASS_ID_BY_NAME[class_name]
    return None, None


def tensor_to_list(value: torch.Tensor) -> List[float]:
    return [round(float(v), 6) for v in value.tolist()]


def make_detections(
    image_width: int,
    image_height: int,
    boxes_cxcywh_norm: torch.Tensor,
    scores: torch.Tensor,
    phrases: List[str],
    alias_index: List[Tuple[str, str]],
    model_name: str,
    image_key: str,
) -> List[Dict]:
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


def load_projection_lookup(dataset_root: Path, projection_summary_path: Optional[Path]) -> Dict[Tuple[int, str], Dict]:
    if not projection_summary_path:
        return {}
    data = read_json(projection_summary_path)
    lookup = {}
    for frame in data.get("frames", []):
        pcd_timestamp = int(frame["pcd_timestamp"])
        for camera, info in frame.get("cameras", {}).items():
            output_image = resolve_dataset_path(dataset_root, info.get("output_image"))
            lookup[(pcd_timestamp, camera)] = {
                "projection_image": str(output_image) if output_image else None,
                "projection": {
                    "boxes_total": info.get("boxes_total"),
                    "boxes_projected": info.get("boxes_projected"),
                    "boxes_skipped_behind_camera": info.get("boxes_skipped_behind_camera"),
                    "boxes_skipped_outside_image": info.get("boxes_skipped_outside_image"),
                    "skipped": info.get("skipped"),
                    "reason": info.get("reason"),
                },
            }
    return lookup


def iter_alignment_records(
    dataset_root: Path,
    aligned_index: Dict,
    projection_lookup: Dict[Tuple[int, str], Dict],
    cameras: List[str],
    valid_time_only: bool,
    max_frames: Optional[int],
) -> Iterable[Dict]:
    frames = aligned_index.get("frames", [])
    if max_frames is not None:
        frames = frames[:max_frames]

    dataset_id = dataset_root.name
    for frame in frames:
        pcd_timestamp = int(frame["pcd_timestamp"])
        frame_valid = bool(frame.get("valid_time_match"))
        for camera in cameras:
            image_info = frame.get("images", {}).get(camera)
            if not image_info:
                continue
            if not image_info.get("exists", True):
                continue
            valid_time_match = bool(image_info.get("valid_time_match"))
            if valid_time_only and not valid_time_match:
                continue

            image_path = resolve_dataset_path(dataset_root, image_info.get("path"))
            if image_path is None:
                continue

            projection_info = projection_lookup.get((pcd_timestamp, camera), {})
            yield {
                "dataset_id": dataset_id,
                "pcd_timestamp": pcd_timestamp,
                "camera": camera,
                "rgb_timestamp": image_info.get("timestamp"),
                "image_delta_ms": image_info.get("delta_ms"),
                "valid_time_match": valid_time_match,
                "alignment_reason": image_info.get("reason"),
                "frame_valid_time_match": frame_valid,
                "frame_reasons": frame.get("reasons", []),
                "rgb_image": str(image_path),
                "rgb_image_rel": image_info.get("path"),
                "projection_image": projection_info.get("projection_image"),
                "projection": projection_info.get("projection"),
                "calib": frame.get("calib"),
                "det_json": frame.get("det_json"),
            }


class GroundingDinoAdapter:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        device: str,
    ):
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

    def predict(self, image_path: Path, image_key: str) -> Dict:
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

    def model_block(self) -> Dict:
        return {
            "name": self.model_name,
            "family": self.model_family,
            "checkpoint": self.checkpoint_name,
            "config": Path(self.config_path).name,
        }

    def params_block(self) -> Dict:
        return {
            "prompt": self.prompt,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
        }


def clone_detection_for_alignment(detection: Dict, pcd_timestamp: int, camera: str, idx: int, model_name: str) -> Dict:
    cloned = dict(detection)
    cloned["det_id"] = f"{model_name}:{pcd_timestamp}:{camera}:{idx}"
    return cloned


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    aligned_index_path = Path(args.aligned_index).expanduser().resolve()
    projection_summary_path = (
        Path(args.projection_summary).expanduser().resolve() if args.projection_summary else None
    )
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = dataset_root / f"aligned_{args.model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    aligned_index = read_json(aligned_index_path)
    projection_lookup = load_projection_lookup(dataset_root, projection_summary_path)
    cameras = args.cameras or aligned_index.get("cameras", [])

    aligned_records = list(
        iter_alignment_records(
            dataset_root=dataset_root,
            aligned_index=aligned_index,
            projection_lookup=projection_lookup,
            cameras=cameras,
            valid_time_only=args.valid_time_only,
            max_frames=args.max_frames,
        )
    )
    if not aligned_records:
        raise RuntimeError("No aligned image records selected.")

    unique_images = {}
    for record in aligned_records:
        unique_images.setdefault(record["rgb_image"], Path(record["rgb_image"]))
    if args.max_images is not None:
        allowed = set(list(unique_images.keys())[: args.max_images])
        aligned_records = [record for record in aligned_records if record["rgb_image"] in allowed]
        unique_images = {key: value for key, value in unique_images.items() if key in allowed}

    adapter = GroundingDinoAdapter(
        config_path=resolve_project_path(args.grounding_config),
        checkpoint_path=resolve_project_path(args.grounding_checkpoint),
        prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
    )

    manifest = {
        "schema_version": "aligned_model_detection_manifest.v1",
        "dataset_root": str(dataset_root),
        "aligned_index": str(aligned_index_path),
        "projection_summary": str(projection_summary_path) if projection_summary_path else None,
        "model": adapter.model_block(),
        "params": adapter.params_block(),
        "classes": CLASS_TABLE,
        "valid_time_only": args.valid_time_only,
        "cameras": cameras,
        "max_frames": args.max_frames,
        "max_images": args.max_images,
    }
    write_json(output_dir / "manifest.json", manifest)

    by_image_path = output_dir / "detections_by_image.jsonl"
    image_results: Dict[str, Dict] = {}
    with by_image_path.open("w", encoding="utf-8") as f:
        for idx, (image_key, image_path) in enumerate(unique_images.items(), start=1):
            result = adapter.predict(image_path=image_path, image_key=Path(image_key).stem)
            image_record = {
                "image_path": image_key,
                "image_id": image_path.stem,
                "camera": image_path.parent.name,
                "image_width": result["image_width"],
                "image_height": result["image_height"],
                "model": adapter.model_block(),
                "params": adapter.params_block(),
                "detections": result["detections"],
            }
            image_results[image_key] = image_record
            f.write(json.dumps(image_record, ensure_ascii=False) + "\n")
            print(f"[image {idx}/{len(unique_images)}] {image_key} detections={len(result['detections'])}")

            if args.save_vis:
                # Save one RGB-timestamp visualization for fast image-level inspection.
                image_vis_dir = output_dir / "vis_by_image" / image_path.parent.name
                image_vis_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_vis_dir / f"{image_path.stem}.jpg"), result["vis"])

    aligned_path = output_dir / "detections_aligned.jsonl"
    total_detections = 0
    with aligned_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(aligned_records, start=1):
            image_record = image_results[record["rgb_image"]]
            detections = [
                clone_detection_for_alignment(
                    detection=det,
                    pcd_timestamp=record["pcd_timestamp"],
                    camera=record["camera"],
                    idx=det_idx,
                    model_name=adapter.model_name,
                )
                for det_idx, det in enumerate(image_record["detections"])
            ]
            aligned_record = {
                "dataset_id": record["dataset_id"],
                "pcd_timestamp": record["pcd_timestamp"],
                "camera": record["camera"],
                "rgb_timestamp": record["rgb_timestamp"],
                "image_delta_ms": record["image_delta_ms"],
                "valid_time_match": record["valid_time_match"],
                "alignment_reason": record["alignment_reason"],
                "frame_valid_time_match": record["frame_valid_time_match"],
                "frame_reasons": record["frame_reasons"],
                "rgb_image": record["rgb_image"],
                "rgb_image_rel": record["rgb_image_rel"],
                "projection_image": record["projection_image"],
                "projection": record["projection"],
                "image_width": image_record["image_width"],
                "image_height": image_record["image_height"],
                "model": adapter.model_block(),
                "params": adapter.params_block(),
                "detections": detections,
            }
            f.write(json.dumps(aligned_record, ensure_ascii=False) + "\n")
            total_detections += len(detections)

            if args.save_vis:
                image_path = Path(record["rgb_image"])
                image_source, _ = load_image(str(image_path))
                if detections:
                    boxes = torch.tensor([det["bbox_cxcywh_norm"] for det in detections], dtype=torch.float32)
                    scores = torch.tensor([det["score"] for det in detections], dtype=torch.float32)
                    phrases = [det["label_raw"] for det in detections]
                    annotated = annotate(image_source, boxes, scores, phrases)
                else:
                    annotated = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
                vis_dir = output_dir / "vis" / record["camera"]
                vis_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(vis_dir / f"{record['pcd_timestamp']}.jpg"), annotated)

            print(
                f"[aligned {idx}/{len(aligned_records)}] "
                f"{record['pcd_timestamp']} {record['camera']} detections={len(detections)}"
            )

    summary = {
        "schema_version": "aligned_model_detection_summary.v1",
        "num_aligned_records": len(aligned_records),
        "num_unique_images": len(unique_images),
        "num_detections_expanded": total_detections,
        "output": {
            "manifest": str(output_dir / "manifest.json"),
            "detections_by_image": str(by_image_path),
            "detections_aligned": str(aligned_path),
        },
        "model": adapter.model_block(),
        "params": adapter.params_block(),
        "classes": CLASS_TABLE,
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Done. Wrote {aligned_path}")


if __name__ == "__main__":
    main()
