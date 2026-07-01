#!/usr/bin/env python3
"""Project 3D detection boxes onto matched camera images.

This final variant keeps the original projection visualization behavior and
adds aligned structured projection output for model comparison.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from render.camera import render_camera_image
from utils.calibration import CameraCalibration, load_box_source_transform, load_calibration
from utils.common import load_json, resolve_path
from utils.detection import load_detection_boxes
from utils.labels import CLASS_TABLE
from utils.roi import (
    ROI_AXIS_INDEX,
    ROI_FRAME,
    RoiConfig,
    empty_roi_counts,
    roi_config_to_dict,
)


DEFAULT_INDEX = Path("2069810204292743169/aligned_index.json")
DEFAULT_OUTPUT_ROOT: Path | None = None
DEFAULT_BOX_SOURCE_FRAME = "lidar_top_GT"
TRANSFORM_CHAIN = "box_source_to_ego_to_camera_to_pixel"


def select_frames(frames: list[dict[str, Any]], *, max_frames: int | None, valid_only: bool) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for frame in frames:
        if valid_only and not frame.get("valid_time_match", False):
            continue
        selected.append(frame)
        if max_frames is not None and len(selected) >= max_frames:
            break
    return selected


def projection_stats(result: dict[str, Any], *, skipped: bool, reason: str | None) -> dict[str, Any]:
    return {
        "boxes_total": result.get("boxes_total"),
        "boxes_projected": result.get("boxes_projected"),
        "boxes_skipped_behind_camera": result.get("boxes_skipped_behind_camera"),
        "boxes_skipped_outside_image": result.get("boxes_skipped_outside_image"),
        "boxes_skipped_unknown_class": result.get("boxes_skipped_unknown_class"),
        "boxes_in_roi": result.get("boxes_in_roi"),
        "boxes_outside_roi": result.get("boxes_outside_roi"),
        "boxes_projected_in_roi": result.get("boxes_projected_in_roi"),
        "boxes_projected_outside_roi": result.get("boxes_projected_outside_roi"),
        "boxes_skipped_outside_roi": result.get("boxes_skipped_outside_roi"),
        "skipped": skipped,
        "reason": reason,
    }


def make_projection_record(
    *,
    dataset_root: Path,
    pcd_timestamp: int,
    frame: dict[str, Any],
    camera_name: str,
    image_info: dict[str, Any],
    image_path: Path | None,
    projection_image: Path | None,
    projection_image_by_image: Path | None,
    image_width: int | None,
    image_height: int | None,
    source: dict[str, Any],
    params: dict[str, Any],
    projection: dict[str, Any],
    projected_boxes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "projection_aligned.v1",
        "dataset_id": dataset_root.name,
        "pcd_timestamp": pcd_timestamp,
        "camera": camera_name,
        "rgb_timestamp": image_info.get("timestamp"),
        "image_delta_ms": image_info.get("delta_ms"),
        "valid_time_match": bool(image_info.get("valid_time_match")),
        "alignment_reason": image_info.get("reason"),
        "frame_valid_time_match": bool(frame.get("valid_time_match")),
        "frame_reasons": frame.get("reasons", []),
        "rgb_image": str(image_path) if image_path else None,
        "rgb_image_rel": image_info.get("path"),
        "projection_image": str(projection_image) if projection_image else None,
        "projection_image_by_image": str(projection_image_by_image) if projection_image_by_image else None,
        "image_width": image_width,
        "image_height": image_height,
        "source": source,
        "params": params,
        "projection": projection,
        "projected_boxes": projected_boxes,
    }


def project_index(
    *,
    index_path: Path,
    output_root: Path | None,
    max_frames: int | None,
    valid_only: bool,
    score_threshold: float,
    box_source_frame: str,
    min_depth: float,
    roi_config: RoiConfig,
    save_debug_vis: bool,
) -> dict[str, Any]:
    index_path = index_path.expanduser().resolve()
    index = load_json(index_path)
    dataset_root = Path(index["dataset_root"]).expanduser()
    if not dataset_root.is_absolute():
        if dataset_root.name == index_path.parent.name:
            dataset_root = index_path.parent
        else:
            dataset_root = (index_path.parent / dataset_root).resolve()
    frames = select_frames(index["frames"], max_frames=max_frames, valid_only=valid_only)
    if output_root is None:
        output_root = dataset_root / "vis_projection_newstruct"

    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(index_path, output_root / "aligned_index.json")

    summary_frames: list[dict[str, Any]] = []
    calibration_cache: dict[tuple[str, str], CameraCalibration] = {}
    box_source_cache: dict[tuple[str, str], np.ndarray] = {}
    total_images = 0
    total_projected = 0
    total_skipped_images = 0
    total_skipped_unknown = 0
    total_boxes_in_roi = 0
    total_boxes_outside_roi = 0
    total_projected_in_roi = 0
    total_projected_outside_roi = 0
    total_skipped_outside_roi = 0
    projections_aligned_path = output_root / "projections_aligned.jsonl"
    params = {
        "score_threshold": score_threshold,
        "min_depth": min_depth,
        "roi": roi_config_to_dict(roi_config),
        "save_debug_vis": save_debug_vis,
    }

    with projections_aligned_path.open("w", encoding="utf-8") as projection_writer:
        for frame in frames:
            pcd_timestamp = int(frame["pcd_timestamp"])
            det_path = resolve_path(dataset_root, frame["det_json"])
            if det_path is None:
                raise RuntimeError(f"missing det_json path for frame {pcd_timestamp}")
            boxes = load_detection_boxes(det_path, score_threshold)

            calib_rel = frame["calib"]["path"]
            calib_path = resolve_path(dataset_root, calib_rel)
            if calib_path is None:
                raise RuntimeError(f"missing calibration path for frame {pcd_timestamp}")

            source = {
                "name": "prelabel-model",
                "family": "projected_3d_detection",
                "det_json": str(det_path),
                "det_json_rel": frame["det_json"],
                "box_source_frame": box_source_frame,
                "transform_chain": TRANSFORM_CHAIN,
            }

            source_cache_key = (str(calib_path), box_source_frame)
            if source_cache_key not in box_source_cache:
                box_source_cache[source_cache_key] = load_box_source_transform(
                    calib_path,
                    box_source_frame,
                )
            t_box_to_ego = box_source_cache[source_cache_key]

            camera_results: dict[str, Any] = {}
            for camera_name, image_info in frame["images"].items():
                image_path = resolve_path(dataset_root, image_info["path"])
                rgb_timestamp = image_info.get("timestamp")
                image_stem = str(rgb_timestamp) if rgb_timestamp is not None else (
                    image_path.stem if image_path else "missing_image"
                )
                output_path = output_root / "vis" / camera_name / f"{pcd_timestamp}.jpg"
                output_by_image_path = output_root / "vis_by_image" / camera_name / f"{image_stem}.jpg"
                debug_output_path = (
                    output_root / "debug_vis" / camera_name / f"{pcd_timestamp}.jpg"
                    if save_debug_vis
                    else None
                )
                if image_path is None:
                    reason = image_info.get("reason") or "image_missing"
                    result = {
                        "input_image": None,
                        "output_image": None,
                        "output_image_by_image": None,
                        "debug_output_image": None,
                        "boxes_total": len(boxes),
                        "boxes_projected": 0,
                        "boxes_skipped_behind_camera": 0,
                        "boxes_skipped_outside_image": 0,
                        "boxes_skipped_unknown_class": 0,
                        **empty_roi_counts(),
                    }
                    result.update(
                        {
                            "image_timestamp": image_info["timestamp"],
                            "image_delta_ms": image_info["delta_ms"],
                            "calib_camera_name": image_info["calib_camera_name"],
                            "skipped": True,
                            "reason": reason,
                        }
                    )
                    camera_results[camera_name] = result
                    projection_record = make_projection_record(
                        dataset_root=dataset_root,
                        pcd_timestamp=pcd_timestamp,
                        frame=frame,
                        camera_name=camera_name,
                        image_info=image_info,
                        image_path=None,
                        projection_image=None,
                        projection_image_by_image=None,
                        image_width=None,
                        image_height=None,
                        source=source,
                        params=params,
                        projection=projection_stats(result, skipped=True, reason=reason),
                        projected_boxes=[],
                    )
                    projection_writer.write(json.dumps(projection_record, ensure_ascii=False) + "\n")
                    total_skipped_images += 1
                    continue
                if not image_path.is_file():
                    reason = "image_file_missing"
                    result = {
                        "input_image": str(image_path),
                        "output_image": None,
                        "output_image_by_image": None,
                        "debug_output_image": None,
                        "boxes_total": len(boxes),
                        "boxes_projected": 0,
                        "boxes_skipped_behind_camera": 0,
                        "boxes_skipped_outside_image": 0,
                        "boxes_skipped_unknown_class": 0,
                        **empty_roi_counts(),
                    }
                    result.update(
                        {
                            "image_timestamp": image_info["timestamp"],
                            "image_delta_ms": image_info["delta_ms"],
                            "calib_camera_name": image_info["calib_camera_name"],
                            "skipped": True,
                            "reason": reason,
                        }
                    )
                    camera_results[camera_name] = result
                    projection_record = make_projection_record(
                        dataset_root=dataset_root,
                        pcd_timestamp=pcd_timestamp,
                        frame=frame,
                        camera_name=camera_name,
                        image_info=image_info,
                        image_path=image_path,
                        projection_image=None,
                        projection_image_by_image=None,
                        image_width=None,
                        image_height=None,
                        source=source,
                        params=params,
                        projection=projection_stats(result, skipped=True, reason=reason),
                        projected_boxes=[],
                    )
                    projection_writer.write(json.dumps(projection_record, ensure_ascii=False) + "\n")
                    total_skipped_images += 1
                    continue

                cache_key = (str(calib_path), camera_name)
                if cache_key not in calibration_cache:
                    calibration_cache[cache_key] = load_calibration(calib_path, camera_name)
                calibration = calibration_cache[cache_key]

                result = render_camera_image(
                    image_path=image_path,
                    output_path=output_path,
                    output_by_image_path=output_by_image_path,
                    debug_output_path=debug_output_path,
                    boxes=boxes,
                    calibration=calibration,
                    t_box_to_ego=t_box_to_ego,
                    min_depth=min_depth,
                    pcd_timestamp=pcd_timestamp,
                    camera_name=camera_name,
                    roi_config=roi_config,
                )
                projected_boxes = result.pop("projected_boxes")
                result.update(
                    {
                        "image_timestamp": image_info["timestamp"],
                        "image_delta_ms": image_info["delta_ms"],
                        "calib_camera_name": image_info["calib_camera_name"],
                        "skipped": False,
                        "reason": None,
                    }
                )
                camera_results[camera_name] = result
                projection_record = make_projection_record(
                    dataset_root=dataset_root,
                    pcd_timestamp=pcd_timestamp,
                    frame=frame,
                    camera_name=camera_name,
                    image_info=image_info,
                    image_path=image_path,
                    projection_image=output_path,
                    projection_image_by_image=output_by_image_path,
                    image_width=result["image_width"],
                    image_height=result["image_height"],
                    source=source,
                    params=params,
                    projection=projection_stats(result, skipped=False, reason=None),
                    projected_boxes=projected_boxes,
                )
                projection_writer.write(json.dumps(projection_record, ensure_ascii=False) + "\n")
                total_images += 1
                total_projected += int(result["boxes_projected"])
                total_skipped_unknown += int(result["boxes_skipped_unknown_class"])
                total_boxes_in_roi += int(result["boxes_in_roi"])
                total_boxes_outside_roi += int(result["boxes_outside_roi"])
                total_projected_in_roi += int(result["boxes_projected_in_roi"])
                total_projected_outside_roi += int(result["boxes_projected_outside_roi"])
                total_skipped_outside_roi += int(result["boxes_skipped_outside_roi"])

            summary_frames.append(
                {
                    "pcd_timestamp": pcd_timestamp,
                    "det_json": frame["det_json"],
                    "calib": frame["calib"],
                    "boxes_total": len(boxes),
                    "cameras": camera_results,
                }
            )

    summary = {
        "schema_version": "projection_summary.v2",
        "index_path": str(index_path),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "valid_only": valid_only,
        "max_frames": max_frames,
        "score_threshold": score_threshold,
        "box_source_frame": box_source_frame,
        "transform_chain": TRANSFORM_CHAIN,
        "min_depth": min_depth,
        "roi": roi_config_to_dict(roi_config),
        "save_debug_vis": save_debug_vis,
        "frame_count": len(summary_frames),
        "image_count": total_images,
        "skipped_image_count": total_skipped_images,
        "total_projected_boxes": total_projected,
        "total_skipped_unknown_class_boxes": total_skipped_unknown,
        "total_boxes_in_roi": total_boxes_in_roi,
        "total_boxes_outside_roi": total_boxes_outside_roi,
        "total_projected_boxes_in_roi": total_projected_in_roi,
        "total_projected_boxes_outside_roi": total_projected_outside_roi,
        "total_skipped_outside_roi": total_skipped_outside_roi,
        "projections_aligned": str(projections_aligned_path),
        "classes": CLASS_TABLE,
        "frames": summary_frames,
    }
    (output_root / "projection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 3D detection boxes onto camera images.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Aligned index JSON path.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output visualization root directory.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=5,
        help="Maximum number of PCD frames to render. Use 0 for all selected frames.",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Also render frames marked invalid by timestamp quality checks.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum detection score to render.",
    )
    parser.add_argument(
        "--box-source-frame",
        default=DEFAULT_BOX_SOURCE_FRAME,
        help=(
            "Coordinate frame of boxes in prelabel-model JSON. "
            "Use 'ego' to skip source->ego transform, or a sensor name from calibration."
        ),
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.1,
        help="Minimum camera-frame Z depth for all 3D box corners.",
    )
    parser.add_argument(
        "--roi-enable",
        dest="roi_enable",
        action="store_true",
        default=True,
        help="Compute ROI metadata for each projected box. Enabled by default in this ROI variant.",
    )
    parser.add_argument(
        "--roi-disable",
        dest="roi_enable",
        action="store_false",
        help="Disable ROI metadata computation.",
    )
    parser.add_argument(
        "--roi-filter",
        action="store_true",
        help="Drop boxes outside ROI before camera projection and JSONL output.",
    )
    parser.add_argument(
        "--save-debug-vis",
        action="store_true",
        help=(
            "Save debug visualizations with all visible projected boxes colored by ROI state "
            "(green=roi, red=out, gray=unknown)."
        ),
    )
    parser.add_argument(
        "--roi-lateral-axis",
        choices=sorted(ROI_AXIS_INDEX),
        default="y",
        help="Ego axis used as lateral left/right direction for ROI checks.",
    )
    parser.add_argument(
        "--roi-longitudinal-axis",
        choices=sorted(ROI_AXIS_INDEX),
        default="x",
        help="Ego axis used as longitudinal front/back direction for ROI checks.",
    )
    parser.add_argument(
        "--roi-lateral-min",
        type=float,
        default=-50.0,
        help="Minimum lateral ROI coordinate in meters.",
    )
    parser.add_argument(
        "--roi-lateral-max",
        type=float,
        default=50.0,
        help="Maximum lateral ROI coordinate in meters.",
    )
    parser.add_argument(
        "--roi-longitudinal-min",
        type=float,
        default=-50.0,
        help="Minimum longitudinal ROI coordinate in meters.",
    )
    parser.add_argument(
        "--roi-longitudinal-max",
        type=float,
        default=150.0,
        help="Maximum longitudinal ROI coordinate in meters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.roi_lateral_axis == args.roi_longitudinal_axis:
        raise ValueError("--roi-lateral-axis and --roi-longitudinal-axis must be different.")
    if args.roi_lateral_min > args.roi_lateral_max:
        raise ValueError("--roi-lateral-min must be <= --roi-lateral-max.")
    if args.roi_longitudinal_min > args.roi_longitudinal_max:
        raise ValueError("--roi-longitudinal-min must be <= --roi-longitudinal-max.")

    roi_config = RoiConfig(
        enabled=bool(args.roi_enable),
        filter_enabled=bool(args.roi_filter),
        frame=ROI_FRAME,
        lateral_axis=args.roi_lateral_axis,
        longitudinal_axis=args.roi_longitudinal_axis,
        lateral_range_m=(args.roi_lateral_min, args.roi_lateral_max),
        longitudinal_range_m=(args.roi_longitudinal_min, args.roi_longitudinal_max),
    )
    max_frames = None if args.max_frames == 0 else args.max_frames
    summary = project_index(
        index_path=args.index,
        output_root=args.output_root,
        max_frames=max_frames,
        valid_only=not args.include_invalid,
        score_threshold=args.score_threshold,
        box_source_frame=args.box_source_frame,
        min_depth=args.min_depth,
        roi_config=roi_config,
        save_debug_vis=args.save_debug_vis,
    )
    print(
        json.dumps(
            {
                "output_root": summary["output_root"],
                "frame_count": summary["frame_count"],
                "image_count": summary["image_count"],
                "skipped_image_count": summary["skipped_image_count"],
                "total_projected_boxes": summary["total_projected_boxes"],
                "box_source_frame": summary["box_source_frame"],
                "transform_chain": summary["transform_chain"],
                "score_threshold": summary["score_threshold"],
                "roi": summary["roi"],
                "save_debug_vis": summary["save_debug_vis"],
                "total_projected_boxes_in_roi": summary["total_projected_boxes_in_roi"],
                "total_projected_boxes_outside_roi": summary["total_projected_boxes_outside_roi"],
                "summary": str(Path(summary["output_root"]) / "projection_summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
