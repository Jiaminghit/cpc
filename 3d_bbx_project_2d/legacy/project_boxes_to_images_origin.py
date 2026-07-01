#!/usr/bin/env python3
"""Project 3D detection boxes onto matched camera images.

This script implements pipeline.md sections 5-10 and supports the section 11
sample validation workflow by defaulting to the first 5 valid PCD frames.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_INDEX = Path("project_task/aligned_index.json")
DEFAULT_OUTPUT_ROOT = Path("2067268107790897153/vis_projection")
DEFAULT_BOX_SOURCE_FRAME = "lidar_top_GT"

BOX_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]

CLASS_COLORS = {
    "car": (255, 80, 40),
    "van": (255, 220, 40),
    "pedestrian": (60, 220, 60),
    "cyclist": (40, 220, 255),
    "traffic_cone": (40, 40, 255),
}
DEFAULT_COLOR = (220, 220, 220)


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    t_ego_to_cam: np.ndarray
    width: int
    height: int
    distortion_model: str


@dataclass(frozen=True)
class DetectionBox:
    object_id: str
    category: str
    score: float
    center: np.ndarray
    size_lwh: np.ndarray
    heading: float


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def resolve_path(dataset_root: Path, rel_path: str | None) -> Path | None:
    if not rel_path:
        return None
    path = Path(rel_path)
    return path if path.is_absolute() else dataset_root / path


def find_sensor(calib_json: dict[str, Any], sensor_name: str) -> dict[str, Any]:
    for sensor in calib_json.get("sensors", []):
        if sensor.get("name") == sensor_name:
            return sensor
    raise KeyError(f"sensor not found in calibration: {sensor_name}")


def load_box_source_transform(calib_path: Path, box_source_frame: str) -> np.ndarray:
    """Return T_box_source_to_ego for the configured source frame."""

    if box_source_frame == "ego":
        return np.eye(4, dtype=np.float64)

    calib_json = load_json(calib_path)
    sensor = find_sensor(calib_json, box_source_frame)
    return np.asarray(sensor["extrinsic"]["transform_matrix_4x4"], dtype=np.float64)


def load_calibration(calib_path: Path, camera_name: str) -> CameraCalibration:
    calib_json = load_json(calib_path)
    sensor = find_sensor(calib_json, camera_name)
    intrinsics = sensor.get("intrinsics") or []
    if not intrinsics:
        raise ValueError(f"camera has no intrinsics: {camera_name}")

    intrinsic = intrinsics[0]
    camera_matrix = np.asarray(intrinsic["camera_matrix"], dtype=np.float64)
    distortion_coeffs = np.asarray(intrinsic.get("distortion_coeffs", []), dtype=np.float64)
    t_cam_to_ego = np.asarray(sensor["extrinsic"]["transform_matrix_4x4"], dtype=np.float64)
    t_ego_to_cam = np.linalg.inv(t_cam_to_ego)

    return CameraCalibration(
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        t_ego_to_cam=t_ego_to_cam,
        width=int(intrinsic["width"]),
        height=int(intrinsic["height"]),
        distortion_model=str(intrinsic.get("distortion_model", "")),
    )


def load_detection_boxes(det_json_path: Path, score_threshold: float) -> list[DetectionBox]:
    det_json = load_json(det_json_path)
    boxes: list[DetectionBox] = []
    for feature in det_json.get("features", []):
        coordinates = feature.get("geometry", {}).get("coordinates", [])
        properties = feature.get("properties", {})
        if len(coordinates) < 3:
            continue

        score = float(properties.get("score", 0.0))
        if score < score_threshold:
            continue

        center = np.asarray(coordinates[0], dtype=np.float64)
        size_lwh = np.asarray(coordinates[1], dtype=np.float64)
        heading = float(coordinates[2][2])
        boxes.append(
            DetectionBox(
                object_id=str(properties.get("id", "")),
                category=str(properties.get("type", "unknown")).lower(),
                score=score,
                center=center,
                size_lwh=size_lwh,
                heading=heading,
            )
        )
    return boxes


def box_to_corners(center: np.ndarray, size_lwh: np.ndarray, heading: float) -> np.ndarray:
    l, w, h = size_lwh.tolist()
    local = np.asarray(
        [
            [l / 2, w / 2, -h / 2],
            [l / 2, -w / 2, -h / 2],
            [-l / 2, -w / 2, -h / 2],
            [-l / 2, w / 2, -h / 2],
            [l / 2, w / 2, h / 2],
            [l / 2, -w / 2, h / 2],
            [-l / 2, -w / 2, h / 2],
            [-l / 2, w / 2, h / 2],
        ],
        dtype=np.float64,
    )
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    rot_z = np.asarray(
        [
            [cos_h, -sin_h, 0.0],
            [sin_h, cos_h, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return local @ rot_z.T + center


def transform_points(points: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
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


def projected_bbox_intersects(points_2d: np.ndarray, width: int, height: int) -> bool:
    min_x = float(np.min(points_2d[:, 0]))
    max_x = float(np.max(points_2d[:, 0]))
    min_y = float(np.min(points_2d[:, 1]))
    max_y = float(np.max(points_2d[:, 1]))
    return max_x >= 0 and max_y >= 0 and min_x < width and min_y < height


def draw_box_edges(
    image: np.ndarray,
    points_2d: np.ndarray,
    label: str,
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    rect = (0, 0, width, height)
    int_points = np.rint(points_2d).astype(np.int32)

    for start, end in BOX_EDGES:
        p1 = tuple(int_points[start].tolist())
        p2 = tuple(int_points[end].tolist())
        ok, clipped_p1, clipped_p2 = cv2.clipLine(rect, p1, p2)
        if ok:
            cv2.line(image, clipped_p1, clipped_p2, color, 2, cv2.LINE_AA)

    label_x = int(np.clip(np.min(points_2d[:, 0]), 0, width - 1))
    label_y = int(np.clip(np.min(points_2d[:, 1]) - 6, 16, height - 1))
    cv2.putText(
        image,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def render_camera_image(
    *,
    image_path: Path,
    output_path: Path,
    boxes: list[DetectionBox],
    calibration: CameraCalibration,
    t_box_to_ego: np.ndarray,
    min_depth: float,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    projected_count = 0
    skipped_behind = 0
    skipped_outside = 0

    for box in boxes:
        corners_box = box_to_corners(box.center, box.size_lwh, box.heading)
        corners_ego = transform_points(corners_box, t_box_to_ego)
        corners_cam = transform_points(corners_ego, calibration.t_ego_to_cam)
        if np.any(corners_cam[:, 2] <= min_depth):
            skipped_behind += 1
            continue

        corners_2d = project_points(corners_cam, calibration)
        if not projected_bbox_intersects(corners_2d, calibration.width, calibration.height):
            skipped_outside += 1
            continue

        color = CLASS_COLORS.get(box.category, DEFAULT_COLOR)
        label = f"{box.category} {box.score:.2f}"
        draw_box_edges(image, corners_2d, label, color)
        projected_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise RuntimeError(f"failed to write image: {output_path}")

    return {
        "input_image": str(image_path),
        "output_image": str(output_path),
        "boxes_total": len(boxes),
        "boxes_projected": projected_count,
        "boxes_skipped_behind_camera": skipped_behind,
        "boxes_skipped_outside_image": skipped_outside,
    }


def select_frames(frames: list[dict[str, Any]], *, max_frames: int | None, valid_only: bool) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for frame in frames:
        if valid_only and not frame.get("valid_time_match", False):
            continue
        selected.append(frame)
        if max_frames is not None and len(selected) >= max_frames:
            break
    return selected


def project_index(
    *,
    index_path: Path,
    output_root: Path,
    max_frames: int | None,
    valid_only: bool,
    score_threshold: float,
    box_source_frame: str,
    min_depth: float,
) -> dict[str, Any]:
    index = load_json(index_path)
    dataset_root = Path(index["dataset_root"])
    frames = select_frames(index["frames"], max_frames=max_frames, valid_only=valid_only)

    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(index_path, output_root / "aligned_index.json")

    summary_frames: list[dict[str, Any]] = []
    calibration_cache: dict[tuple[str, str], CameraCalibration] = {}
    box_source_cache: dict[tuple[str, str], np.ndarray] = {}
    total_images = 0
    total_projected = 0

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
            if image_path is None:
                raise RuntimeError(f"missing image path for {pcd_timestamp} {camera_name}")

            cache_key = (str(calib_path), camera_name)
            if cache_key not in calibration_cache:
                calibration_cache[cache_key] = load_calibration(calib_path, camera_name)
            calibration = calibration_cache[cache_key]

            output_path = output_root / camera_name / f"{pcd_timestamp}.jpg"
            result = render_camera_image(
                image_path=image_path,
                output_path=output_path,
                boxes=boxes,
                calibration=calibration,
                t_box_to_ego=t_box_to_ego,
                min_depth=min_depth,
            )
            result.update(
                {
                    "image_timestamp": image_info["timestamp"],
                    "image_delta_ms": image_info["delta_ms"],
                    "calib_camera_name": image_info["calib_camera_name"],
                }
            )
            camera_results[camera_name] = result
            total_images += 1
            total_projected += int(result["boxes_projected"])

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
        "schema_version": "projection_summary.v1",
        "index_path": str(index_path),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "valid_only": valid_only,
        "max_frames": max_frames,
        "score_threshold": score_threshold,
        "box_source_frame": box_source_frame,
        "transform_chain": "box_source_to_ego_to_camera_to_pixel",
        "min_depth": min_depth,
        "frame_count": len(summary_frames),
        "image_count": total_images,
        "total_projected_boxes": total_projected,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_frames = None if args.max_frames == 0 else args.max_frames
    summary = project_index(
        index_path=args.index,
        output_root=args.output_root,
        max_frames=max_frames,
        valid_only=not args.include_invalid,
        score_threshold=args.score_threshold,
        box_source_frame=args.box_source_frame,
        min_depth=args.min_depth,
    )
    print(
        json.dumps(
            {
                "output_root": summary["output_root"],
                "frame_count": summary["frame_count"],
                "image_count": summary["image_count"],
                "total_projected_boxes": summary["total_projected_boxes"],
                "box_source_frame": summary["box_source_frame"],
                "transform_chain": summary["transform_chain"],
                "score_threshold": summary["score_threshold"],
                "summary": str(Path(summary["output_root"]) / "projection_summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
