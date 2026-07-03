from __future__ import annotations

"""基于 LiDAR 点云的 2D 检测 ROI 过滤 pipeline。

输入是 `detections_aligned.jsonl`。流程会为每个 2D bbox 查找匹配 PCD，
把 LiDAR 点投影到相机图像，估计 bbox 内目标的 ego 坐标代表点，
再判断是否落在配置的 ego ROI 内。
"""

import argparse
import copy
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.cache import cache_put, get_cached
from utils.calibration import CameraCalibration, load_camera_calibration, load_source_to_ego
from utils.geometry import ProjectedPointCloud, project_lidar_to_camera
from utils.io import load_json, read_jsonl, resolve_path, write_json
from utils.pcd import read_pcd_xyz
from utils.roi import (
    DetectionPointSelection,
    ROI_FRAME,
    RoiConfig,
    empty_selection,
    estimate_detection_position,
    roi_config_to_dict,
    unknown_roi,
)
from utils.summary import ratio, summarize_points, update_counter_block
from utils.visualization import (
    POINT_IN_BBOX_COLOR,
    POINT_USED_COLOR,
    draw_detection,
    draw_points,
)


def validate_args(args: argparse.Namespace) -> None:
    """校验 ROI 与点选择相关参数，尽早暴露无效配置。"""
    if args.roi_lateral_axis == args.roi_longitudinal_axis:
        raise ValueError("--roi-lateral-axis and --roi-longitudinal-axis must be different.")
    if args.roi_lateral_min > args.roi_lateral_max:
        raise ValueError("--roi-lateral-min must be <= --roi-lateral-max.")
    if args.roi_longitudinal_min > args.roi_longitudinal_max:
        raise ValueError("--roi-longitudinal-min must be <= --roi-longitudinal-max.")
    if not 0.0 < args.depth_percentile <= 1.0:
        raise ValueError("--depth-percentile must be in (0, 1].")
    if not 0.0 <= args.bbox_candidate_y_min_ratio < args.bbox_candidate_y_max_ratio <= 1.0:
        raise ValueError(
            "--bbox-candidate-y-min-ratio and --bbox-candidate-y-max-ratio must satisfy "
            "0 <= min < max <= 1."
        )
    if args.min_points_in_bbox <= 0 or args.min_points_used <= 0:
        raise ValueError("--min-points-in-bbox and --min-points-used must be positive.")


def build_roi_config(args: argparse.Namespace) -> RoiConfig:
    """将 CLI 中零散的 ROI 参数整理成不可变配置对象。"""
    return RoiConfig(
        enabled=True,
        frame=ROI_FRAME,
        lateral_axis=args.roi_lateral_axis,
        longitudinal_axis=args.roi_longitudinal_axis,
        lateral_range_m=(args.roi_lateral_min, args.roi_lateral_max),
        longitudinal_range_m=(args.roi_longitudinal_min, args.roi_longitudinal_max),
    )


def record_frame_valid(record: dict[str, Any], frames_by_timestamp: dict[int, dict[str, Any]]) -> bool:
    """判断一条检测记录所属 PCD frame 是否通过整帧时间同步检查。"""
    if "frame_valid_time_match" in record:
        return bool(record.get("frame_valid_time_match"))
    frame = frames_by_timestamp.get(int(record["pcd_timestamp"]))
    return bool(frame and frame.get("valid_time_match"))


def run_lidar_roi_filter(args: argparse.Namespace) -> None:
    """执行完整的 LiDAR ROI 过滤流程。"""
    dataset_root = args.dataset_root.expanduser().resolve()
    aligned_index_path = (
        args.aligned_index.expanduser().resolve()
        if args.aligned_index
        else dataset_root / "aligned_index.json"
    )
    detections_path = (
        args.detections_jsonl.expanduser().resolve()
        if args.detections_jsonl
        else dataset_root / "aligned_grounding_dino_b" / "detections_aligned.jsonl"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else dataset_root / "aligned_grounding_dino_b_roi"
    )

    validate_args(args)
    roi_config = build_roi_config(args)

    # aligned_index 提供 pcd/calib/rgb 路径；detections_jsonl 提供已对齐的 2D 检测框。
    aligned_index = load_json(aligned_index_path)
    frames_by_timestamp = {
        int(frame["pcd_timestamp"]): frame for frame in aligned_index.get("frames", [])
    }
    records = read_jsonl(detections_path)
    if args.cameras:
        camera_filter = set(args.cameras)
        records = [record for record in records if record.get("camera") in camera_filter]
    if not args.include_invalid:
        records = [record for record in records if record_frame_valid(record, frames_by_timestamp)]
    if args.max_records is not None:
        records = records[: args.max_records]

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / "detections_aligned_with_roi.jsonl"
    roi_only_path = output_dir / "detections_aligned_roi_only.jsonl"

    manifest = {
        "schema_version": "dino_lidar_roi_filter_manifest.v1",
        "dataset_root": str(dataset_root),
        "aligned_index": str(aligned_index_path),
        "detections_jsonl": str(detections_path),
        "output_dir": str(output_dir),
        "params": {
            "box_source_frame": args.box_source_frame,
            "min_depth": args.min_depth,
            "min_points_in_bbox": args.min_points_in_bbox,
            "min_points_used": args.min_points_used,
            "depth_percentile": args.depth_percentile,
            "bbox_candidate_y_min_ratio": args.bbox_candidate_y_min_ratio,
            "bbox_candidate_y_max_ratio": args.bbox_candidate_y_max_ratio,
            "roi": roi_config_to_dict(roi_config),
            "include_invalid": bool(args.include_invalid),
            "valid_frame_only": not bool(args.include_invalid),
            "cameras": args.cameras,
            "max_records": args.max_records,
            "save_vis": args.save_vis,
            "save_debug_vis": args.save_debug_vis,
        },
    }
    write_json(output_dir / "manifest.json", manifest)

    # 这些缓存避免同一 PCD、同一标定、同一相机投影在多条记录间重复计算。
    pcd_cache: OrderedDict[int, np.ndarray] = OrderedDict()
    calib_cache: OrderedDict[Path, dict[str, Any]] = OrderedDict()
    source_cache: OrderedDict[tuple[Path, str], np.ndarray] = OrderedDict()
    camera_calib_cache: OrderedDict[tuple[Path, str], CameraCalibration] = OrderedDict()
    projection_cache: OrderedDict[tuple[int, str, int, int], ProjectedPointCloud] = OrderedDict()

    summary: dict[str, Any] = {
        "schema_version": "dino_lidar_roi_filter_summary.v1",
        "num_records": 0,
        "num_records_without_pcd": 0,
        "num_records_without_calib": 0,
        "num_records_without_projected_points": 0,
        "num_detections_total": 0,
        "num_detections_in_roi": 0,
        "num_detections_outside_roi": 0,
        "num_detections_unknown_roi": 0,
        "unknown_reasons": {},
        "by_camera": {},
        "by_class": {},
        "points_in_bbox": [],
        "points_used": [],
        "outputs": {
            "manifest": str(output_dir / "manifest.json"),
            "detections_aligned_with_roi": str(annotated_path),
            "detections_aligned_roi_only": str(roi_only_path),
            "roi_filter_summary": str(output_dir / "roi_filter_summary.json"),
        },
    }

    with annotated_path.open("w", encoding="utf-8") as annotated_writer, roi_only_path.open(
        "w", encoding="utf-8"
    ) as roi_only_writer:
        for record_idx, record in enumerate(records, start=1):
            # 每条 record 对应一个 PCD timestamp + camera 的检测结果。
            pcd_timestamp = int(record["pcd_timestamp"])
            camera = str(record["camera"])
            frame = frames_by_timestamp.get(pcd_timestamp)
            detections = record.get("detections", [])
            annotated_record = copy.deepcopy(record)
            annotated_detections = annotated_record.get("detections", [])
            debug_selections: list[tuple[dict[str, Any], DetectionPointSelection]] = []

            record_reason: str | None = None
            projected: ProjectedPointCloud | None = None

            if annotated_detections:
                if frame is None:
                    record_reason = "missing_aligned_frame"
                else:
                    # 找到该 PCD-camera 记录对应的点云和标定文件。
                    pcd_path = resolve_path(dataset_root, frame.get("pcd", {}).get("path"))
                    calib_path = resolve_path(dataset_root, frame.get("calib", {}).get("path"))
                    if pcd_path is None or not pcd_path.is_file():
                        record_reason = "missing_pcd"
                        summary["num_records_without_pcd"] += 1
                    elif calib_path is None or not calib_path.is_file():
                        record_reason = "missing_calib"
                        summary["num_records_without_calib"] += 1
                    else:
                        image_width = int(record.get("image_width") or 0)
                        image_height = int(record.get("image_height") or 0)
                        if image_width <= 0 or image_height <= 0:
                            image_info = frame.get("images", {}).get(camera, {})
                            rgb_path = resolve_path(dataset_root, image_info.get("path"))
                            if rgb_path and rgb_path.is_file():
                                image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                                if image is not None:
                                    image_height, image_width = image.shape[:2]
                        projection_key = (pcd_timestamp, camera, image_width, image_height)
                        projected = get_cached(projection_cache, projection_key)
                        if projected is None:
                            # 缓存 miss 时才读取标定、点云并完成 LiDAR->image 投影。
                            calib_json = get_cached(calib_cache, calib_path)
                            if calib_json is None:
                                calib_json = load_json(calib_path)
                                cache_put(calib_cache, calib_path, calib_json, max_items=8)

                            source_key = (calib_path, args.box_source_frame)
                            t_lidar_to_ego = get_cached(source_cache, source_key)
                            if t_lidar_to_ego is None:
                                t_lidar_to_ego = load_source_to_ego(calib_json, args.box_source_frame)
                                cache_put(source_cache, source_key, t_lidar_to_ego, max_items=8)

                            camera_key = (calib_path, camera)
                            calibration = get_cached(camera_calib_cache, camera_key)
                            if calibration is None:
                                calibration = load_camera_calibration(calib_json, camera)
                                cache_put(camera_calib_cache, camera_key, calibration, max_items=24)

                            points_lidar = get_cached(pcd_cache, pcd_timestamp)
                            if points_lidar is None:
                                points_lidar = read_pcd_xyz(pcd_path)
                                cache_put(pcd_cache, pcd_timestamp, points_lidar, args.pcd_cache_size)

                            if image_width <= 0:
                                image_width = calibration.width
                            if image_height <= 0:
                                image_height = calibration.height
                            projected = project_lidar_to_camera(
                                points_lidar=points_lidar,
                                t_lidar_to_ego=t_lidar_to_ego,
                                calibration=calibration,
                                image_width=image_width,
                                image_height=image_height,
                                min_depth=args.min_depth,
                            )
                            cache_put(
                                projection_cache,
                                projection_key,
                                projected,
                                args.projection_cache_size,
                            )
                        if projected.pixels.shape[0] == 0:
                            record_reason = "no_projected_lidar_points"
                            summary["num_records_without_projected_points"] += 1

            for detection in annotated_detections:
                # 对每个 bbox 估计一个 ego 代表点，并写入 detection["lidar_roi"]。
                if record_reason:
                    roi = unknown_roi(record_reason)
                    selection = empty_selection()
                else:
                    assert projected is not None
                    roi, selection = estimate_detection_position(
                        detection=detection,
                        projected=projected,
                        roi_config=roi_config,
                        min_points_in_bbox=args.min_points_in_bbox,
                        min_points_used=args.min_points_used,
                        depth_percentile=args.depth_percentile,
                        bbox_candidate_y_min_ratio=args.bbox_candidate_y_min_ratio,
                        bbox_candidate_y_max_ratio=args.bbox_candidate_y_max_ratio,
                    )
                detection["lidar_roi"] = roi
                debug_selections.append((detection, selection))

                class_name = detection.get("class_name") or detection.get("label_raw") or "unknown"
                summary["num_detections_total"] += 1
                if roi.get("in_roi") is True:
                    summary["num_detections_in_roi"] += 1
                elif roi.get("in_roi") is False:
                    summary["num_detections_outside_roi"] += 1
                else:
                    summary["num_detections_unknown_roi"] += 1
                    reason = roi.get("reason") or "unknown"
                    summary["unknown_reasons"][reason] = summary["unknown_reasons"].get(reason, 0) + 1
                summary["points_in_bbox"].append(int(roi.get("points_in_bbox") or 0))
                summary["points_used"].append(int(roi.get("points_used") or 0))

                camera_block = summary["by_camera"].setdefault(camera, {})
                update_counter_block(camera_block, roi, class_name)
                class_block = summary["by_class"].setdefault(class_name, {})
                update_counter_block(class_block, roi, None)

            roi_only_record = copy.deepcopy(annotated_record)
            # ROI-only 文件只保留明确落在 ROI 内的检测；unknown/outside 都会被过滤。
            roi_only_record["detections"] = [
                detection
                for detection in roi_only_record.get("detections", [])
                if detection.get("lidar_roi", {}).get("in_roi") is True
            ]
            annotated_writer.write(json.dumps(annotated_record, ensure_ascii=False) + "\n")
            roi_only_writer.write(json.dumps(roi_only_record, ensure_ascii=False) + "\n")
            summary["num_records"] += 1

            if args.save_vis or args.save_debug_vis:
                # 普通可视化只画 ROI 内目标；debug 可视化额外画 bbox 内点和代表点候选。
                image_path = Path(str(record.get("rgb_image") or ""))
                if not image_path.is_file():
                    image_path = resolve_path(dataset_root, record.get("rgb_image_rel")) or image_path
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path.is_file() else None
                if image is not None:
                    if args.save_vis:
                        vis = image.copy()
                        for detection in annotated_detections:
                            if detection.get("lidar_roi", {}).get("in_roi") is True:
                                draw_detection(
                                    vis,
                                    detection,
                                    color_mode="class",
                                    include_roi_status=False,
                                )
                        vis_dir = output_dir / "vis" / camera
                        vis_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(vis_dir / f"{pcd_timestamp}.jpg"), vis)

                        rgb_timestamp = record.get("rgb_timestamp")
                        image_stem = str(rgb_timestamp) if rgb_timestamp is not None else image_path.stem
                        vis_by_image_dir = output_dir / "vis_by_image" / camera
                        vis_by_image_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(vis_by_image_dir / f"{image_stem}.jpg"), vis)

                    if args.save_debug_vis:
                        debug = image.copy()
                        for detection, selection in debug_selections:
                            draw_detection(debug, detection)
                            draw_points(debug, selection.in_bbox_pixels, POINT_IN_BBOX_COLOR, 1)
                            draw_points(debug, selection.used_pixels, POINT_USED_COLOR, 2)
                        debug_dir = output_dir / "debug_vis" / camera
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(debug_dir / f"{pcd_timestamp}.jpg"), debug)

            print(
                f"[{record_idx}/{len(records)}] {pcd_timestamp} {camera} "
                f"detections={len(detections)} roi={len(roi_only_record['detections'])}"
            )

    total = int(summary["num_detections_total"])
    # 最终 summary 补充比例和点数分布，便于检查过滤是否过严或点云过稀。
    summary["roi_ratios"] = {
        "in_roi": ratio(int(summary["num_detections_in_roi"]), total),
        "outside_roi": ratio(int(summary["num_detections_outside_roi"]), total),
        "unknown_roi": ratio(int(summary["num_detections_unknown_roi"]), total),
    }
    summary["points_in_bbox_stats"] = summarize_points(summary.pop("points_in_bbox"))
    summary["points_used_stats"] = summarize_points(summary.pop("points_used"))
    summary["params"] = manifest["params"]
    write_json(output_dir / "roi_filter_summary.json", summary)
    print(f"Done. Wrote {annotated_path}")
