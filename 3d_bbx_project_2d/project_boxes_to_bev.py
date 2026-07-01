#!/usr/bin/env python3
"""Render prelabel-model 3D detection boxes in BEV view."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from render.bev import (
    BevConfig,
    PointCloudConfig,
    bev_config_to_dict,
    point_cloud_config_to_dict,
    render_bev_frame,
)
from utils.calibration import load_box_source_transform
from utils.common import load_json, resolve_path
from utils.detection import load_detection_boxes
from utils.labels import CLASS_TABLE
from utils.roi import (
    ROI_AXIS_INDEX,
    ROI_FRAME,
    RoiConfig,
    roi_config_to_dict,
)


DEFAULT_INDEX = Path("/home/c64508/桌面/dataset/2069758074335653889/aligned_index.json")
DEFAULT_OUTPUT_ROOT: Path | None = None
DEFAULT_BOX_SOURCE_FRAME = "lidar_top_GT"
TRANSFORM_CHAIN = "box_source_to_ego_to_bev_pixel"


def select_frames(frames: list[dict[str, Any]], *, max_frames: int | None, valid_only: bool) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for frame in frames:
        if valid_only and not frame.get("valid_time_match", False):
            continue
        selected.append(frame)
        if max_frames is not None and len(selected) >= max_frames:
            break
    return selected


def resolve_dataset_root(index_path: Path, index: dict[str, Any]) -> Path:
    dataset_root = Path(index["dataset_root"]).expanduser()
    if not dataset_root.is_absolute():
        if dataset_root.name == index_path.parent.name:
            dataset_root = index_path.parent
        else:
            dataset_root = (index_path.parent / dataset_root).resolve()
    return dataset_root


def path_for_output(path: Path, dataset_root: Path) -> str:
    try:
        return str(path.relative_to(dataset_root))
    except ValueError:
        return str(path)


def maybe_calibration_latest(dataset_root: Path) -> str | None:
    candidate = dataset_root / "struct_json" / "calibration_latest.json"
    if candidate.is_file():
        return path_for_output(candidate, dataset_root)
    return None


def build_frames_from_pcd_meta(dataset_root: Path, max_frames: int | None) -> list[dict[str, Any]]:
    pcd_meta = load_json(dataset_root / "pcd" / "meta.json")
    calib_rel = maybe_calibration_latest(dataset_root)
    frames: list[dict[str, Any]] = []
    for frame in pcd_meta.get("frame_map", []):
        det_rel = frame.get("preann", {}).get("prelabel-model")
        if not det_rel:
            continue
        timestamp = int(frame["timestamp"])
        pcd_file = frame.get("files", {}).get("p128_0")
        frames.append(
            {
                "pcd_timestamp": timestamp,
                "pcd": {
                    "timestamp": timestamp,
                    "path": f"pcd/{pcd_file}" if pcd_file else None,
                    "exists": bool(pcd_file),
                },
                "det_json": f"pcd/{det_rel}",
                "calib": {
                    "path": calib_rel,
                    "timestamp": None,
                    "delta_ms": None,
                    "valid_time_match": calib_rel is not None,
                    "exists": calib_rel is not None,
                    "reason": None if calib_rel is not None else "calibration_latest_missing",
                },
                "valid_time_match": True,
                "reasons": [],
            }
        )
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames


def build_frames_from_prelabel_dir(prelabel_dir: Path, max_frames: int | None) -> tuple[Path, list[dict[str, Any]]]:
    prelabel_dir = prelabel_dir.expanduser().resolve()
    if prelabel_dir.name != "prelabel-model" or prelabel_dir.parent.name != "pcd":
        raise ValueError("--prelabel-dir is expected to point to <dataset_root>/pcd/prelabel-model")
    dataset_root = prelabel_dir.parent.parent
    calib_rel = maybe_calibration_latest(dataset_root)
    frames: list[dict[str, Any]] = []
    for det_path in sorted(prelabel_dir.glob("*.json")):
        timestamp = int(det_path.stem)
        det_rel = path_for_output(det_path, dataset_root)
        pcd_path = dataset_root / "pcd" / "p128_0" / f"{timestamp}.pcd"
        frames.append(
            {
                "pcd_timestamp": timestamp,
                "pcd": {
                    "timestamp": timestamp,
                    "path": path_for_output(pcd_path, dataset_root),
                    "exists": pcd_path.is_file(),
                },
                "det_json": det_rel,
                "calib": {
                    "path": calib_rel,
                    "timestamp": None,
                    "delta_ms": None,
                    "valid_time_match": calib_rel is not None,
                    "exists": calib_rel is not None,
                    "reason": None if calib_rel is not None else "calibration_latest_missing",
                },
                "valid_time_match": True,
                "reasons": [],
            }
        )
        if max_frames is not None and len(frames) >= max_frames:
            break
    return dataset_root, frames


def load_input_frames(
    *,
    index_path: Path | None,
    dataset_root_arg: Path | None,
    prelabel_dir: Path | None,
    max_frames: int | None,
    valid_only: bool,
) -> tuple[Path, list[dict[str, Any]], Path | None]:
    if index_path is not None:
        index_path = index_path.expanduser().resolve()
        index = load_json(index_path)
        dataset_root = resolve_dataset_root(index_path, index)
        frames = select_frames(index["frames"], max_frames=max_frames, valid_only=valid_only)
        return dataset_root, frames, index_path

    if prelabel_dir is not None:
        dataset_root, frames = build_frames_from_prelabel_dir(prelabel_dir, max_frames)
        return dataset_root, frames, None

    if dataset_root_arg is None:
        raise ValueError("One of --index, --dataset-root, or --prelabel-dir must be provided.")

    dataset_root = dataset_root_arg.expanduser().resolve()
    frames = build_frames_from_pcd_meta(dataset_root, max_frames)
    return dataset_root, frames, None


def make_bev_config(args: argparse.Namespace) -> BevConfig:
    if args.x_min >= args.x_max:
        raise ValueError("--x-min must be < --x-max.")
    if args.y_min >= args.y_max:
        raise ValueError("--y-min must be < --y-max.")
    if args.meters_per_pixel <= 0:
        raise ValueError("--meters-per-pixel must be > 0.")
    if args.grid_interval_m <= 0:
        raise ValueError("--grid-interval-m must be > 0.")
    y_positive_direction = getattr(args, "bev_y_positive_direction", "left")
    if y_positive_direction not in {"left", "right"}:
        raise ValueError("--bev-y-positive-direction must be 'left' or 'right'.")

    width = args.bev_width or int(math.ceil((args.y_max - args.y_min) / args.meters_per_pixel)) + 1
    height = args.bev_height or int(math.ceil((args.x_max - args.x_min) / args.meters_per_pixel)) + 1
    if width <= 1 or height <= 1:
        raise ValueError("BEV canvas width and height must be greater than 1.")

    return BevConfig(
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        meters_per_pixel=args.meters_per_pixel,
        width=width,
        height=height,
        y_positive_direction=y_positive_direction,
        draw_labels=bool(args.draw_labels),
        draw_heading=bool(args.draw_heading),
        grid_interval_m=args.grid_interval_m,
    )


def make_point_cloud_config(args: argparse.Namespace) -> PointCloudConfig:
    if args.point_z_min > args.point_z_max:
        raise ValueError("--point-z-min must be <= --point-z-max.")
    if args.point_size < 1:
        raise ValueError("--point-size must be >= 1.")
    return PointCloudConfig(
        enabled=bool(args.draw_points),
        source_frame=args.pcd_source_frame,
        color_mode=args.point_color_mode,
        z_min=args.point_z_min,
        z_max=args.point_z_max,
        point_size=args.point_size,
    )


def make_bev_record(
    *,
    dataset_root: Path,
    pcd_timestamp: int,
    frame: dict[str, Any],
    det_path: Path,
    pcd_path: Path | None,
    bev_image: Path,
    source: dict[str, Any],
    params: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "bev_projection.v1",
        "dataset_id": dataset_root.name,
        "pcd_timestamp": pcd_timestamp,
        "det_json": str(det_path),
        "det_json_rel": frame.get("det_json"),
        "pcd_path": str(pcd_path) if pcd_path else None,
        "pcd_path_rel": frame.get("pcd", {}).get("path"),
        "bev_image": str(bev_image),
        "source": source,
        "params": params,
        "projection": {
            "boxes_total": result["boxes_total"],
            "boxes_drawn": result["boxes_drawn"],
            "boxes_skipped_unknown_class": result["boxes_skipped_unknown_class"],
            "boxes_skipped_outside_canvas": result["boxes_skipped_outside_canvas"],
            "boxes_in_roi": result["boxes_in_roi"],
            "boxes_outside_roi": result["boxes_outside_roi"],
            "boxes_projected_in_roi": result["boxes_projected_in_roi"],
            "boxes_projected_outside_roi": result["boxes_projected_outside_roi"],
            "boxes_skipped_outside_roi": result["boxes_skipped_outside_roi"],
            "point_cloud": result["point_cloud"],
        },
        "frame_valid_time_match": bool(frame.get("valid_time_match", True)),
        "frame_reasons": frame.get("reasons", []),
        "calib": frame.get("calib"),
        "boxes": result["boxes"],
    }


def project_to_bev(
    *,
    index_path: Path | None,
    dataset_root_arg: Path | None,
    prelabel_dir: Path | None,
    output_root: Path | None,
    max_frames: int | None,
    valid_only: bool,
    score_threshold: float,
    box_source_frame: str,
    bev_config: BevConfig,
    roi_config: RoiConfig,
    point_config: PointCloudConfig,
) -> dict[str, Any]:
    dataset_root, frames, resolved_index_path = load_input_frames(
        index_path=index_path,
        dataset_root_arg=dataset_root_arg,
        prelabel_dir=prelabel_dir,
        max_frames=max_frames,
        valid_only=valid_only,
    )

    if output_root is None:
        output_root = dataset_root / "vis_bev_prelabel_model"
    output_root.mkdir(parents=True, exist_ok=True)
    if resolved_index_path is not None:
        shutil.copy2(resolved_index_path, output_root / "aligned_index.json")

    bev_jsonl_path = output_root / "bev_boxes.jsonl"
    summary_frames: list[dict[str, Any]] = []
    box_source_cache: dict[tuple[str, str], np.ndarray] = {}
    pcd_source_cache: dict[tuple[str, str], np.ndarray] = {}
    total_drawn = 0
    total_skipped_unknown = 0
    total_skipped_outside_canvas = 0
    total_pcd_points = 0
    total_pcd_points_valid = 0
    total_pcd_points_drawn = 0
    total_boxes_in_roi = 0
    total_boxes_outside_roi = 0
    total_projected_in_roi = 0
    total_projected_outside_roi = 0
    total_skipped_outside_roi = 0

    params = {
        "score_threshold": score_threshold,
        "box_source_frame": box_source_frame,
        "transform_chain": TRANSFORM_CHAIN,
        "bev": bev_config_to_dict(bev_config),
        "roi": roi_config_to_dict(roi_config),
        "point_cloud": point_cloud_config_to_dict(point_config),
    }

    with bev_jsonl_path.open("w", encoding="utf-8") as writer:
        for frame in frames:
            pcd_timestamp = int(frame["pcd_timestamp"])
            det_path = resolve_path(dataset_root, frame.get("det_json"))
            if det_path is None:
                raise RuntimeError(f"missing det_json path for frame {pcd_timestamp}")
            boxes = load_detection_boxes(det_path, score_threshold)
            pcd_path = resolve_path(dataset_root, frame.get("pcd", {}).get("path"))

            calib_rel = frame.get("calib", {}).get("path")
            calib_path = resolve_path(dataset_root, calib_rel)
            if box_source_frame != "ego" and calib_path is None:
                raise RuntimeError(
                    f"frame {pcd_timestamp} has no calibration path; use --index or --box-source-frame ego"
                )
            if point_config.enabled and point_config.source_frame != "ego" and calib_path is None:
                raise RuntimeError(
                    f"frame {pcd_timestamp} has no calibration path for point cloud; use --index or --pcd-source-frame ego"
                )

            source = {
                "name": "prelabel-model",
                "family": "bev_3d_detection",
                "det_json": str(det_path),
                "det_json_rel": frame.get("det_json"),
                "box_source_frame": box_source_frame,
                "transform_chain": TRANSFORM_CHAIN,
            }

            source_cache_key = (str(calib_path), box_source_frame)
            if source_cache_key not in box_source_cache:
                if calib_path is None:
                    box_source_cache[source_cache_key] = np.eye(4, dtype=np.float64)
                else:
                    box_source_cache[source_cache_key] = load_box_source_transform(calib_path, box_source_frame)
            t_box_to_ego = box_source_cache[source_cache_key]

            pcd_cache_key = (str(calib_path), point_config.source_frame)
            if pcd_cache_key not in pcd_source_cache:
                if calib_path is None:
                    pcd_source_cache[pcd_cache_key] = np.eye(4, dtype=np.float64)
                else:
                    pcd_source_cache[pcd_cache_key] = load_box_source_transform(calib_path, point_config.source_frame)
            t_pcd_to_ego = pcd_source_cache[pcd_cache_key]

            bev_image = output_root / "bev" / f"{pcd_timestamp}.jpg"
            result = render_bev_frame(
                output_path=bev_image,
                boxes=boxes,
                t_box_to_ego=t_box_to_ego,
                pcd_path=pcd_path,
                t_pcd_to_ego=t_pcd_to_ego,
                pcd_timestamp=pcd_timestamp,
                bev_config=bev_config,
                roi_config=roi_config,
                point_config=point_config,
            )

            record = make_bev_record(
                dataset_root=dataset_root,
                pcd_timestamp=pcd_timestamp,
                frame=frame,
                det_path=det_path,
                pcd_path=pcd_path,
                bev_image=bev_image,
                source=source,
                params=params,
                result=result,
            )
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_drawn += int(result["boxes_drawn"])
            total_skipped_unknown += int(result["boxes_skipped_unknown_class"])
            total_skipped_outside_canvas += int(result["boxes_skipped_outside_canvas"])
            total_pcd_points += int(result["point_cloud"]["points_total"])
            total_pcd_points_valid += int(result["point_cloud"]["points_valid"])
            total_pcd_points_drawn += int(result["point_cloud"]["points_drawn"])
            total_boxes_in_roi += int(result["boxes_in_roi"])
            total_boxes_outside_roi += int(result["boxes_outside_roi"])
            total_projected_in_roi += int(result["boxes_projected_in_roi"])
            total_projected_outside_roi += int(result["boxes_projected_outside_roi"])
            total_skipped_outside_roi += int(result["boxes_skipped_outside_roi"])
            summary_frames.append(
                {
                    "pcd_timestamp": pcd_timestamp,
                    "det_json": frame.get("det_json"),
                    "pcd": frame.get("pcd"),
                    "calib": frame.get("calib"),
                    "bev_image": str(bev_image),
                    "boxes_total": result["boxes_total"],
                    "boxes_drawn": result["boxes_drawn"],
                    "boxes_skipped_unknown_class": result["boxes_skipped_unknown_class"],
                    "boxes_skipped_outside_canvas": result["boxes_skipped_outside_canvas"],
                    "boxes_in_roi": result["boxes_in_roi"],
                    "boxes_outside_roi": result["boxes_outside_roi"],
                    "boxes_projected_in_roi": result["boxes_projected_in_roi"],
                    "boxes_projected_outside_roi": result["boxes_projected_outside_roi"],
                    "boxes_skipped_outside_roi": result["boxes_skipped_outside_roi"],
                    "point_cloud": result["point_cloud"],
                }
            )

    summary = {
        "schema_version": "bev_projection_summary.v1",
        "index_path": str(resolved_index_path) if resolved_index_path else None,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "valid_only": valid_only,
        "max_frames": max_frames,
        "score_threshold": score_threshold,
        "box_source_frame": box_source_frame,
        "transform_chain": TRANSFORM_CHAIN,
        "bev": bev_config_to_dict(bev_config),
        "roi": roi_config_to_dict(roi_config),
        "point_cloud": point_cloud_config_to_dict(point_config),
        "frame_count": len(summary_frames),
        "total_drawn_boxes": total_drawn,
        "total_skipped_unknown_class_boxes": total_skipped_unknown,
        "total_skipped_outside_canvas_boxes": total_skipped_outside_canvas,
        "total_pcd_points": total_pcd_points,
        "total_pcd_points_valid": total_pcd_points_valid,
        "total_pcd_points_drawn": total_pcd_points_drawn,
        "total_boxes_in_roi": total_boxes_in_roi,
        "total_boxes_outside_roi": total_boxes_outside_roi,
        "total_projected_boxes_in_roi": total_projected_in_roi,
        "total_projected_boxes_outside_roi": total_projected_outside_roi,
        "total_skipped_outside_roi": total_skipped_outside_roi,
        "bev_boxes": str(bev_jsonl_path),
        "classes": CLASS_TABLE,
        "frames": summary_frames,
    }
    (output_root / "bev_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render 3D detection boxes in BEV view.")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--index",
        type=Path,
        help=f"Aligned index JSON path. Defaults to {DEFAULT_INDEX} when no input source is provided.",
    )
    input_group.add_argument("--dataset-root", type=Path, help="Dataset root containing pcd/meta.json.")
    input_group.add_argument("--prelabel-dir", type=Path, help="Path to <dataset_root>/pcd/prelabel-model.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output BEV visualization root directory.",
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
        help="Also render frames marked invalid by timestamp quality checks when using --index.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Minimum detection score to render.")
    parser.add_argument(
        "--box-source-frame",
        default=DEFAULT_BOX_SOURCE_FRAME,
        help="Coordinate frame of boxes. Use 'ego' to skip source->ego transform.",
    )
    parser.add_argument("--bev-width", type=int, help="BEV image width in pixels.")
    parser.add_argument("--bev-height", type=int, help="BEV image height in pixels.")
    parser.add_argument("--meters-per-pixel", type=float, default=0.2, help="BEV resolution used when width/height are omitted.")
    parser.add_argument("--x-min", type=float, default=-60.0, help="Minimum ego x coordinate shown in BEV.")
    parser.add_argument("--x-max", type=float, default=160.0, help="Maximum ego x coordinate shown in BEV.")
    parser.add_argument("--y-min", type=float, default=-60.0, help="Minimum ego y coordinate shown in BEV.")
    parser.add_argument("--y-max", type=float, default=60.0, help="Maximum ego y coordinate shown in BEV.")
    parser.add_argument(
        "--bev-y-positive-direction",
        choices=["left", "right"],
        default="left",
        help="Image side for positive ego y in BEV. Use 'right' to reproduce the previous behavior.",
    )
    parser.add_argument("--grid-interval-m", type=float, default=10.0, help="BEV grid interval in meters.")
    parser.add_argument("--draw-labels", dest="draw_labels", action="store_true", default=True, help="Draw class and score labels.")
    parser.add_argument("--no-draw-labels", dest="draw_labels", action="store_false", help="Do not draw labels.")
    parser.add_argument("--draw-heading", dest="draw_heading", action="store_true", default=True, help="Draw heading arrows.")
    parser.add_argument("--no-draw-heading", dest="draw_heading", action="store_false", help="Do not draw heading arrows.")
    parser.add_argument("--draw-points", dest="draw_points", action="store_true", default=True, help="Draw PCD point cloud as BEV background.")
    parser.add_argument("--no-draw-points", dest="draw_points", action="store_false", help="Do not draw PCD point cloud.")
    parser.add_argument(
        "--pcd-source-frame",
        default=DEFAULT_BOX_SOURCE_FRAME,
        help="Coordinate frame of PCD points. Use 'ego' to skip source->ego transform.",
    )
    parser.add_argument(
        "--point-color-mode",
        choices=["height", "intensity", "constant"],
        default="height",
        help="How to color BEV point cloud pixels.",
    )
    parser.add_argument("--point-z-min", type=float, default=-5.0, help="Minimum ego z for points drawn in BEV.")
    parser.add_argument("--point-z-max", type=float, default=5.0, help="Maximum ego z for points drawn in BEV.")
    parser.add_argument("--point-size", type=int, default=1, help="Point size in BEV pixels.")
    parser.add_argument(
        "--roi-enable",
        dest="roi_enable",
        action="store_true",
        default=True,
        help="Compute ROI metadata. Enabled by default.",
    )
    parser.add_argument("--roi-disable", dest="roi_enable", action="store_false", help="Disable ROI metadata.")
    parser.add_argument("--roi-filter", action="store_true", help="Drop boxes outside ROI before BEV output.")
    parser.add_argument("--roi-lateral-axis", choices=sorted(ROI_AXIS_INDEX), default="y", help="Ego axis used as lateral direction.")
    parser.add_argument("--roi-longitudinal-axis", choices=sorted(ROI_AXIS_INDEX), default="x", help="Ego axis used as longitudinal direction.")
    parser.add_argument("--roi-lateral-min", type=float, default=-50.0, help="Minimum lateral ROI coordinate in meters.")
    parser.add_argument("--roi-lateral-max", type=float, default=50.0, help="Maximum lateral ROI coordinate in meters.")
    parser.add_argument("--roi-longitudinal-min", type=float, default=-50.0, help="Minimum longitudinal ROI coordinate in meters.")
    parser.add_argument("--roi-longitudinal-max", type=float, default=150.0, help="Maximum longitudinal ROI coordinate in meters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.roi_lateral_axis == args.roi_longitudinal_axis:
        raise ValueError("--roi-lateral-axis and --roi-longitudinal-axis must be different.")
    if args.roi_lateral_min > args.roi_lateral_max:
        raise ValueError("--roi-lateral-min must be <= --roi-lateral-max.")
    if args.roi_longitudinal_min > args.roi_longitudinal_max:
        raise ValueError("--roi-longitudinal-min must be <= --roi-longitudinal-max.")

    max_frames = None if args.max_frames == 0 else args.max_frames
    bev_config = make_bev_config(args)
    point_config = make_point_cloud_config(args)
    roi_config = RoiConfig(
        enabled=bool(args.roi_enable),
        filter_enabled=bool(args.roi_filter),
        frame=ROI_FRAME,
        lateral_axis=args.roi_lateral_axis,
        longitudinal_axis=args.roi_longitudinal_axis,
        lateral_range_m=(args.roi_lateral_min, args.roi_lateral_max),
        longitudinal_range_m=(args.roi_longitudinal_min, args.roi_longitudinal_max),
    )

    summary = project_to_bev(
        index_path=args.index if args.index is not None else (
            DEFAULT_INDEX if args.dataset_root is None and args.prelabel_dir is None else None
        ),
        dataset_root_arg=args.dataset_root,
        prelabel_dir=args.prelabel_dir,
        output_root=args.output_root,
        max_frames=max_frames,
        valid_only=not args.include_invalid,
        score_threshold=args.score_threshold,
        box_source_frame=args.box_source_frame,
        bev_config=bev_config,
        roi_config=roi_config,
        point_config=point_config,
    )
    print(
        json.dumps(
            {
                "output_root": summary["output_root"],
                "frame_count": summary["frame_count"],
                "total_drawn_boxes": summary["total_drawn_boxes"],
                "total_skipped_unknown_class_boxes": summary["total_skipped_unknown_class_boxes"],
                "total_skipped_outside_canvas_boxes": summary["total_skipped_outside_canvas_boxes"],
                "total_pcd_points_drawn": summary["total_pcd_points_drawn"],
                "box_source_frame": summary["box_source_frame"],
                "point_cloud": summary["point_cloud"],
                "roi": summary["roi"],
                "bev_summary": str(Path(summary["output_root"]) / "bev_summary.json"),
                "bev_boxes": summary["bev_boxes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
