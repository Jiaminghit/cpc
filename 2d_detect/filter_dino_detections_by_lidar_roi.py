#!/usr/bin/env python3
"""Filter aligned GroundingDINO detections by LiDAR-derived ego ROI.

This script keeps the original aligned DINO records, projects the matched PCD
points into each camera image, estimates an ego-frame representative point for
each 2D detection bbox, and writes both ROI-annotated and ROI-only JSONL files.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_BOX_SOURCE_FRAME = "lidar_top_GT"
ROI_FRAME = "ego"
ROI_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
METHOD_NAME = "project_lidar_points_to_image_bbox_depth_percentile"

ROI_IN_COLOR = (60, 220, 60)
ROI_OUT_COLOR = (40, 40, 255)
ROI_UNKNOWN_COLOR = (170, 170, 170)
POINT_IN_BBOX_COLOR = (255, 180, 40)
POINT_USED_COLOR = (40, 220, 255)
CLASS_COLORS = {
    "Car": (255, 80, 40),
    "Van": (255, 220, 40),
    "Pedestrian": (60, 220, 60),
    "Cyclist": (40, 220, 255),
    "Traffic_cone": (40, 40, 255),
}
DEFAULT_CLASS_COLOR = (220, 220, 220)


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    t_ego_to_cam: np.ndarray
    width: int
    height: int
    distortion_model: str


@dataclass(frozen=True)
class RoiConfig:
    enabled: bool
    frame: str
    lateral_axis: str
    longitudinal_axis: str
    lateral_range_m: tuple[float, float]
    longitudinal_range_m: tuple[float, float]


@dataclass(frozen=True)
class ProjectedPointCloud:
    pixels: np.ndarray
    points_camera: np.ndarray
    points_ego: np.ndarray


@dataclass(frozen=True)
class DetectionPointSelection:
    points_in_bbox: int
    points_used: int
    in_bbox_pixels: np.ndarray
    used_pixels: np.ndarray
    used_camera: np.ndarray
    used_ego: np.ndarray


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def round_float(value: float) -> float:
    return round(float(value), 6)


def array_to_rounded_list(values: np.ndarray | None) -> list[Any] | None:
    if values is None:
        return None
    return np.round(values.astype(np.float64), 6).tolist()


def resolve_path(dataset_root: Path, rel_path: str | None) -> Path | None:
    if not rel_path:
        return None
    path = Path(rel_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == dataset_root.name:
        return dataset_root.parent / path
    return dataset_root / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sensor_name(sensor: dict[str, Any]) -> str:
    return str(sensor.get("sensor_name") or sensor.get("name") or "<unknown>")


def find_sensor(calib_json: dict[str, Any], name: str) -> dict[str, Any]:
    for sensor in calib_json.get("sensors", []):
        if sensor.get("sensor_name") == name or sensor.get("name") == name:
            return sensor
    raise KeyError(f"sensor not found in calibration: {name}")


def sensor_transform_matrix(sensor: dict[str, Any]) -> np.ndarray:
    extrinsics = sensor.get("extrinsics") or sensor.get("extrinsic") or {}
    if "transform_matrix_4x4" in extrinsics:
        return np.asarray(extrinsics["transform_matrix_4x4"], dtype=np.float64)

    transform_matrix = extrinsics.get("transform_matrix")
    if transform_matrix is None:
        raise KeyError(f"sensor has no transform_matrix: {sensor_name(sensor)}")

    matrix = np.asarray(transform_matrix, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f"transform_matrix must have 16 values for {sensor_name(sensor)}")
    return matrix.reshape(4, 4)


def intrinsic_camera_matrix(intrinsic: dict[str, Any]) -> np.ndarray:
    if "camera_matrix" in intrinsic:
        return np.asarray(intrinsic["camera_matrix"], dtype=np.float64)
    return np.asarray(
        [
            [float(intrinsic["fx"]), 0.0, float(intrinsic["cx"])],
            [0.0, float(intrinsic["fy"]), float(intrinsic["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def load_source_to_ego(calib_json: dict[str, Any], source_frame: str) -> np.ndarray:
    if source_frame == "ego":
        return np.eye(4, dtype=np.float64)
    sensor = find_sensor(calib_json, source_frame)
    return sensor_transform_matrix(sensor)


def load_camera_calibration(calib_json: dict[str, Any], camera_name: str) -> CameraCalibration:
    sensor = find_sensor(calib_json, camera_name)
    intrinsics = sensor.get("intrinsics") or []
    if not intrinsics:
        raise ValueError(f"camera has no intrinsics: {camera_name}")

    intrinsic = intrinsics[0]
    camera_matrix = intrinsic_camera_matrix(intrinsic)
    distortion_coeffs = np.asarray(intrinsic.get("distortion_coeffs", []), dtype=np.float64)
    t_cam_to_ego = sensor_transform_matrix(sensor)
    return CameraCalibration(
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        t_ego_to_cam=np.linalg.inv(t_cam_to_ego),
        width=int(intrinsic["width"]),
        height=int(intrinsic["height"]),
        distortion_model=str(intrinsic.get("distortion_model", "")),
    )


def transform_points(points: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    transformed = homogeneous @ transform_4x4.T
    return transformed[:, :3]


def project_points(points_cam: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    object_points = points_cam.reshape(-1, 1, 3).astype(np.float64)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)

    if calibration.distortion_model.lower() == "fisheye":
        distortion = calibration.distortion_coeffs[:4].reshape(4, 1)
        image_points, _ = cv2.fisheye.projectPoints(
            object_points,
            rvec,
            tvec,
            calibration.camera_matrix,
            distortion,
        )
    else:
        image_points, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            calibration.camera_matrix,
            calibration.distortion_coeffs,
        )
    return image_points.reshape(-1, 2)


def pcd_numpy_type(field_type: str, size: int) -> str:
    if field_type == "F":
        if size == 4:
            return "<f4"
        if size == 8:
            return "<f8"
    if field_type == "I":
        if size in {1, 2, 4, 8}:
            return f"<i{size}"
    if field_type == "U":
        if size in {1, 2, 4, 8}:
            return f"<u{size}"
    raise ValueError(f"unsupported PCD field type/size: {field_type}{size}")


def parse_pcd_header(path: Path) -> tuple[dict[str, Any], int]:
    header: dict[str, Any] = {}
    with path.open("rb") as reader:
        while True:
            line = reader.readline()
            if not line:
                raise ValueError(f"PCD file has no DATA line: {path}")
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded or decoded.startswith("#"):
                continue
            parts = decoded.split()
            key = parts[0].upper()
            values = parts[1:]
            header[key] = values
            if key == "DATA":
                return header, reader.tell()


def read_pcd_xyz(path: Path) -> np.ndarray:
    header, data_offset = parse_pcd_header(path)
    fields = header.get("FIELDS")
    sizes = [int(value) for value in header.get("SIZE", [])]
    types = header.get("TYPE")
    counts = [int(value) for value in header.get("COUNT", [])] if "COUNT" in header else None
    data_kind = (header.get("DATA") or [""])[0].lower()
    points_count = int((header.get("POINTS") or [0])[0])

    if not fields or not sizes or not types:
        raise ValueError(f"PCD header missing FIELDS/SIZE/TYPE: {path}")
    if counts is None:
        counts = [1] * len(fields)
    if not {"x", "y", "z"}.issubset(set(fields)):
        raise ValueError(f"PCD file must contain x/y/z fields: {path}")

    if data_kind == "ascii":
        data = np.loadtxt(path, comments="#", skiprows=len(header) + 1)
        field_index = {name: idx for idx, name in enumerate(fields)}
        return np.asarray(
            data[:, [field_index["x"], field_index["y"], field_index["z"]]],
            dtype=np.float64,
        )

    if data_kind != "binary":
        raise ValueError(f"unsupported PCD DATA format: {data_kind}")

    dtype_fields: list[tuple[str, Any]] = []
    for name, field_type, size, count in zip(fields, types, sizes, counts):
        base_dtype = np.dtype(pcd_numpy_type(field_type, size))
        if count == 1:
            dtype_fields.append((name, base_dtype))
        else:
            dtype_fields.append((name, base_dtype, (count,)))
    dtype = np.dtype(dtype_fields)

    with path.open("rb") as reader:
        reader.seek(data_offset)
        raw = reader.read()
    data = np.frombuffer(raw, dtype=dtype, count=points_count)
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64, copy=False)
    finite_mask = np.isfinite(xyz).all(axis=1)
    return xyz[finite_mask]


def roi_config_to_dict(roi_config: RoiConfig) -> dict[str, Any]:
    return {
        "enabled": roi_config.enabled,
        "frame": roi_config.frame,
        "lateral_axis": roi_config.lateral_axis,
        "longitudinal_axis": roi_config.longitudinal_axis,
        "lateral_range_m": [
            round_float(roi_config.lateral_range_m[0]),
            round_float(roi_config.lateral_range_m[1]),
        ],
        "longitudinal_range_m": [
            round_float(roi_config.longitudinal_range_m[0]),
            round_float(roi_config.longitudinal_range_m[1]),
        ],
    }


def unknown_roi(reason: str, *, points_in_bbox: int = 0, points_used: int = 0) -> dict[str, Any]:
    return {
        "enabled": True,
        "in_roi": None,
        "reason": reason,
        "frame": ROI_FRAME,
        "center_ego": None,
        "representative_point_ego": None,
        "representative_point_camera": None,
        "depth_camera_m": None,
        "points_in_bbox": int(points_in_bbox),
        "points_used": int(points_used),
        "method": METHOD_NAME,
    }


def build_roi_info(
    *,
    representative_point_ego: np.ndarray,
    representative_point_camera: np.ndarray,
    points_in_bbox: int,
    points_used: int,
    roi_config: RoiConfig,
) -> dict[str, Any]:
    lateral_idx = ROI_AXIS_INDEX[roi_config.lateral_axis]
    longitudinal_idx = ROI_AXIS_INDEX[roi_config.longitudinal_axis]
    lateral = float(representative_point_ego[lateral_idx])
    longitudinal = float(representative_point_ego[longitudinal_idx])
    lateral_min, lateral_max = roi_config.lateral_range_m
    longitudinal_min, longitudinal_max = roi_config.longitudinal_range_m
    in_roi = (
        lateral_min <= lateral <= lateral_max
        and longitudinal_min <= longitudinal <= longitudinal_max
    )
    return {
        "enabled": True,
        "in_roi": bool(in_roi),
        "reason": None,
        "frame": roi_config.frame,
        "center_ego": array_to_rounded_list(representative_point_ego),
        "representative_point_ego": array_to_rounded_list(representative_point_ego),
        "representative_point_camera": array_to_rounded_list(representative_point_camera),
        "depth_camera_m": round_float(representative_point_camera[2]),
        "points_in_bbox": int(points_in_bbox),
        "points_used": int(points_used),
        "lateral_axis": roi_config.lateral_axis,
        "longitudinal_axis": roi_config.longitudinal_axis,
        "lateral_m": round_float(lateral),
        "longitudinal_m": round_float(longitudinal),
        "lateral_range_m": [round_float(lateral_min), round_float(lateral_max)],
        "longitudinal_range_m": [round_float(longitudinal_min), round_float(longitudinal_max)],
        "method": METHOD_NAME,
    }


def estimate_detection_position(
    *,
    detection: dict[str, Any],
    projected: ProjectedPointCloud,
    roi_config: RoiConfig,
    min_points_in_bbox: int,
    min_points_used: int,
    depth_percentile: float,
) -> tuple[dict[str, Any], DetectionPointSelection]:
    bbox = detection.get("bbox_xyxy")
    if not bbox or len(bbox) != 4:
        empty = np.empty((0, 2), dtype=np.float64)
        empty3 = np.empty((0, 3), dtype=np.float64)
        return unknown_roi("missing_bbox_xyxy"), DetectionPointSelection(0, 0, empty, empty, empty3, empty3)

    x1, y1, x2, y2 = [float(value) for value in bbox]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    pixels = projected.pixels
    in_bbox_mask = (
        (pixels[:, 0] >= x1)
        & (pixels[:, 0] <= x2)
        & (pixels[:, 1] >= y1)
        & (pixels[:, 1] <= y2)
    )
    in_bbox_indices = np.flatnonzero(in_bbox_mask)
    points_in_bbox = int(in_bbox_indices.size)
    if points_in_bbox < min_points_in_bbox:
        empty3 = np.empty((0, 3), dtype=np.float64)
        return (
            unknown_roi("insufficient_lidar_points", points_in_bbox=points_in_bbox),
            DetectionPointSelection(
                points_in_bbox,
                0,
                projected.pixels[in_bbox_indices],
                np.empty((0, 2), dtype=np.float64),
                empty3,
                empty3,
            ),
        )

    candidate_camera = projected.points_camera[in_bbox_indices]
    depths = candidate_camera[:, 2]
    order = np.argsort(depths)
    percentile_count = int(math.ceil(points_in_bbox * depth_percentile))
    use_count = max(min_points_used, percentile_count)
    use_count = min(points_in_bbox, use_count)
    used_local_indices = order[:use_count]
    used_indices = in_bbox_indices[used_local_indices]

    used_ego = projected.points_ego[used_indices]
    used_camera = projected.points_camera[used_indices]
    representative_ego = np.median(used_ego, axis=0)
    representative_camera = np.median(used_camera, axis=0)
    roi = build_roi_info(
        representative_point_ego=representative_ego,
        representative_point_camera=representative_camera,
        points_in_bbox=points_in_bbox,
        points_used=int(use_count),
        roi_config=roi_config,
    )
    return (
        roi,
        DetectionPointSelection(
            points_in_bbox,
            int(use_count),
            projected.pixels[in_bbox_indices],
            projected.pixels[used_indices],
            used_camera,
            used_ego,
        ),
    )


def project_lidar_to_camera(
    *,
    points_lidar: np.ndarray,
    t_lidar_to_ego: np.ndarray,
    calibration: CameraCalibration,
    image_width: int,
    image_height: int,
    min_depth: float,
) -> ProjectedPointCloud:
    points_ego = transform_points(points_lidar, t_lidar_to_ego)
    points_camera = transform_points(points_ego, calibration.t_ego_to_cam)
    depth_mask = points_camera[:, 2] > min_depth
    points_ego = points_ego[depth_mask]
    points_camera = points_camera[depth_mask]
    if points_camera.size == 0:
        empty2 = np.empty((0, 2), dtype=np.float64)
        empty3 = np.empty((0, 3), dtype=np.float64)
        return ProjectedPointCloud(empty2, empty3, empty3)

    pixels = project_points(points_camera, calibration)
    finite_mask = np.isfinite(pixels).all(axis=1)
    image_mask = (
        finite_mask
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < image_width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < image_height)
    )
    return ProjectedPointCloud(
        pixels=pixels[image_mask],
        points_camera=points_camera[image_mask],
        points_ego=points_ego[image_mask],
    )


def cache_put(cache: OrderedDict[Any, Any], key: Any, value: Any, max_items: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_items:
        cache.popitem(last=False)


def get_cached(cache: OrderedDict[Any, Any], key: Any) -> Any | None:
    if key not in cache:
        return None
    cache.move_to_end(key)
    return cache[key]


def color_for_roi(roi: dict[str, Any]) -> tuple[int, int, int]:
    if roi.get("in_roi") is True:
        return ROI_IN_COLOR
    if roi.get("in_roi") is False:
        return ROI_OUT_COLOR
    return ROI_UNKNOWN_COLOR


def color_for_class(detection: dict[str, Any]) -> tuple[int, int, int]:
    class_name = detection.get("class_name")
    return CLASS_COLORS.get(class_name, DEFAULT_CLASS_COLOR)


def format_roi_position(roi: dict[str, Any]) -> str:
    longitudinal = roi.get("longitudinal_m")
    lateral = roi.get("lateral_m")
    if isinstance(longitudinal, (int, float)) and isinstance(lateral, (int, float)):
        return f"long={float(longitudinal):.2f}m lat={float(lateral):.2f}m"
    return "long=na lat=na"


def draw_detection(
    image: np.ndarray,
    detection: dict[str, Any],
    *,
    draw_unknown: bool = True,
    color_mode: str = "roi",
    include_roi_status: bool = True,
) -> None:
    roi = detection.get("lidar_roi", {})
    if roi.get("in_roi") is None and not draw_unknown:
        return
    bbox = detection.get("bbox_xyxy")
    if not bbox or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    height, width = image.shape[:2]
    x1 = int(np.clip(x1, 0, width - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    y2 = int(np.clip(y2, 0, height - 1))
    color = color_for_class(detection) if color_mode == "class" else color_for_roi(roi)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    class_name = detection.get("class_name") or detection.get("label_raw") or "unknown"
    score = detection.get("score")
    if include_roi_status:
        status = "roi" if roi.get("in_roi") is True else "out" if roi.get("in_roi") is False else "unknown"
        position = format_roi_position(roi)
        label = (
            f"{class_name} {score:.2f} {status} {position}"
            if isinstance(score, (int, float))
            else f"{class_name} {status} {position}"
        )
    else:
        label = f"{class_name} {score:.2f}" if isinstance(score, (int, float)) else str(class_name)
    label_y = int(np.clip(y1 - 6, 16, height - 1))
    cv2.putText(
        image,
        label,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_points(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
    if points.size == 0:
        return
    height, width = image.shape[:2]
    rounded = np.rint(points).astype(np.int32)
    max_points = min(2000, rounded.shape[0])
    if rounded.shape[0] > max_points:
        step = max(1, rounded.shape[0] // max_points)
        rounded = rounded[::step]
    for x, y in rounded:
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def update_counter_block(block: dict[str, Any], roi: dict[str, Any], class_name: str | None = None) -> None:
    block["detections_total"] = block.get("detections_total", 0) + 1
    if roi.get("in_roi") is True:
        block["detections_in_roi"] = block.get("detections_in_roi", 0) + 1
    elif roi.get("in_roi") is False:
        block["detections_outside_roi"] = block.get("detections_outside_roi", 0) + 1
    else:
        block["detections_unknown_roi"] = block.get("detections_unknown_roi", 0) + 1
        reason = roi.get("reason") or "unknown"
        reasons = block.setdefault("unknown_reasons", {})
        reasons[reason] = reasons.get(reason, 0) + 1
    if class_name is not None:
        classes = block.setdefault("by_class", {})
        class_block = classes.setdefault(class_name, {})
        update_counter_block(class_block, roi, None)


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round_float(numerator / denominator)


def summarize_points(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": int(np.min(arr)),
        "median": round_float(np.median(arr)),
        "max": int(np.max(arr)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter aligned GroundingDINO detections using LiDAR-derived ego ROI."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--aligned-index", type=Path, default=None)
    parser.add_argument("--detections-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--box-source-frame", default=DEFAULT_BOX_SOURCE_FRAME)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--min-points-in-bbox", type=int, default=3)
    parser.add_argument("--min-points-used", type=int, default=3)
    parser.add_argument("--depth-percentile", type=float, default=0.2)
    parser.add_argument("--roi-lateral-axis", choices=sorted(ROI_AXIS_INDEX), default="y")
    parser.add_argument("--roi-longitudinal-axis", choices=sorted(ROI_AXIS_INDEX), default="x")
    parser.add_argument("--roi-lateral-min", type=float, default=-50.0)
    parser.add_argument("--roi-lateral-max", type=float, default=50.0)
    parser.add_argument("--roi-longitudinal-min", type=float, default=-50.0)
    parser.add_argument("--roi-longitudinal-max", type=float, default=150.0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--valid-time-only", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--save-debug-vis", action="store_true")
    parser.add_argument("--pcd-cache-size", type=int, default=2)
    parser.add_argument("--projection-cache-size", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    if args.roi_lateral_axis == args.roi_longitudinal_axis:
        raise ValueError("--roi-lateral-axis and --roi-longitudinal-axis must be different.")
    if args.roi_lateral_min > args.roi_lateral_max:
        raise ValueError("--roi-lateral-min must be <= --roi-lateral-max.")
    if args.roi_longitudinal_min > args.roi_longitudinal_max:
        raise ValueError("--roi-longitudinal-min must be <= --roi-longitudinal-max.")
    if not 0.0 < args.depth_percentile <= 1.0:
        raise ValueError("--depth-percentile must be in (0, 1].")
    if args.min_points_in_bbox <= 0 or args.min_points_used <= 0:
        raise ValueError("--min-points-in-bbox and --min-points-used must be positive.")

    roi_config = RoiConfig(
        enabled=True,
        frame=ROI_FRAME,
        lateral_axis=args.roi_lateral_axis,
        longitudinal_axis=args.roi_longitudinal_axis,
        lateral_range_m=(args.roi_lateral_min, args.roi_lateral_max),
        longitudinal_range_m=(args.roi_longitudinal_min, args.roi_longitudinal_max),
    )

    aligned_index = load_json(aligned_index_path)
    frames_by_timestamp = {
        int(frame["pcd_timestamp"]): frame for frame in aligned_index.get("frames", [])
    }
    records = read_jsonl(detections_path)
    if args.cameras:
        camera_filter = set(args.cameras)
        records = [record for record in records if record.get("camera") in camera_filter]
    if args.valid_time_only:
        records = [record for record in records if record.get("valid_time_match")]
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
            "roi": roi_config_to_dict(roi_config),
            "valid_time_only": args.valid_time_only,
            "cameras": args.cameras,
            "max_records": args.max_records,
            "save_vis": args.save_vis,
            "save_debug_vis": args.save_debug_vis,
        },
    }
    write_json(output_dir / "manifest.json", manifest)

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

            for det_idx, detection in enumerate(annotated_detections):
                if record_reason:
                    roi = unknown_roi(record_reason)
                    empty = np.empty((0, 2), dtype=np.float64)
                    empty3 = np.empty((0, 3), dtype=np.float64)
                    selection = DetectionPointSelection(0, 0, empty, empty, empty3, empty3)
                else:
                    assert projected is not None
                    roi, selection = estimate_detection_position(
                        detection=detection,
                        projected=projected,
                        roi_config=roi_config,
                        min_points_in_bbox=args.min_points_in_bbox,
                        min_points_used=args.min_points_used,
                        depth_percentile=args.depth_percentile,
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
            roi_only_record["detections"] = [
                detection
                for detection in roi_only_record.get("detections", [])
                if detection.get("lidar_roi", {}).get("in_roi") is True
            ]
            annotated_writer.write(json.dumps(annotated_record, ensure_ascii=False) + "\n")
            roi_only_writer.write(json.dumps(roi_only_record, ensure_ascii=False) + "\n")
            summary["num_records"] += 1

            if args.save_vis or args.save_debug_vis:
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


if __name__ == "__main__":
    main()
