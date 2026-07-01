#!/usr/bin/env python3
"""Build timestamp alignment index for 3D box projection.

This script implements pipeline.md sections 1-4:
- use PCD/prelabel frames as the primary timeline
- nearest-neighbor match calibration frames
- nearest-neighbor match the available confirmed camera image streams
- record timing deltas and validity flags
"""

from __future__ import annotations

import argparse
import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_DATASET_ROOT = Path("2067268107790897153")
DEFAULT_OUTPUT = Path("project_task/aligned_index.json")

CAMERAS = [
    "camera_front_wide",
    "camera_rear",
    "camera_side_left_front",
    "camera_side_left_rear",
    "camera_side_right_front",
    "camera_side_right_rear",
]

SKIPPED_CAMERAS = {
    "camera_front_long": "calibration camera mapping is not confirmed",
    "camera_front_narrow": "image camera mapping is not confirmed",
}


@dataclass(frozen=True)
class TimedPath:
    timestamp: int
    path: str


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def index_path(child_path: str) -> str:
    return str(Path(child_path))


def build_pcd_entries(dataset_root: Path, pcd_meta: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for frame in pcd_meta.get("frame_map", []):
        ts = int(frame["timestamp"])
        pcd_file = frame.get("files", {}).get("p128_0")
        det_file = frame.get("preann", {}).get("prelabel-model")
        entries.append(
            {
                "timestamp": ts,
                "pcd_path": index_path(f"pcd/{pcd_file}") if pcd_file else None,
                "det_json": index_path(f"pcd/{det_file}") if det_file else None,
            }
        )
    return sorted(entries, key=lambda item: item["timestamp"])


def build_single_stream(
    dataset_root: Path,
    meta: dict[str, Any],
    *,
    file_key: str,
    base_dir: str,
) -> list[TimedPath]:
    stream: list[TimedPath] = []
    for frame in meta.get("frame_map", []):
        ts = int(frame["timestamp"])
        file_path = frame.get("files", {}).get(file_key)
        if file_path:
            stream.append(TimedPath(ts, index_path(f"{base_dir}/{file_path}")))
    return sorted(stream, key=lambda item: item.timestamp)


def build_camera_streams(dataset_root: Path, jpg_meta: dict[str, Any]) -> dict[str, list[TimedPath]]:
    streams: dict[str, list[TimedPath]] = {camera: [] for camera in CAMERAS}
    for frame in jpg_meta.get("frame_map", []):
        ts = int(frame["timestamp"])
        for camera_name, file_path in frame.get("files", {}).items():
            if camera_name in streams:
                streams[camera_name].append(
                    TimedPath(ts, index_path(f"jpg/{file_path}"))
                )
    for camera_name in streams:
        streams[camera_name].sort(key=lambda item: item.timestamp)
    return streams


def available_cameras(camera_streams: dict[str, list[TimedPath]]) -> list[str]:
    return [camera_name for camera_name in CAMERAS if camera_streams.get(camera_name)]


def skipped_unavailable_cameras(camera_streams: dict[str, list[TimedPath]]) -> dict[str, str]:
    return {
        camera_name: "image stream is not present in jpg meta"
        for camera_name in CAMERAS
        if not camera_streams.get(camera_name)
    }


def nearest(stream: list[TimedPath], timestamp: int) -> tuple[TimedPath | None, int | None]:
    if not stream:
        return None, None

    timestamps = [item.timestamp for item in stream]
    idx = bisect.bisect_left(timestamps, timestamp)
    candidates: list[TimedPath] = []
    if idx > 0:
        candidates.append(stream[idx - 1])
    if idx < len(stream):
        candidates.append(stream[idx])

    best = min(candidates, key=lambda item: abs(item.timestamp - timestamp))
    return best, abs(best.timestamp - timestamp)


def delta_ms(delta_ns: int | None) -> float | None:
    return None if delta_ns is None else delta_ns / 1_000_000.0


def path_exists(dataset_root: Path, path: str | None) -> bool:
    if not path:
        return False
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.is_file()
    return (dataset_root / candidate).is_file()


def match_quality(delta: float | None, threshold: float) -> tuple[bool, str | None]:
    if delta is None:
        return False, "missing_match"
    if delta > threshold:
        return False, "delta_ms_exceeds_threshold"
    return True, None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * pct) - 1))
    return ordered[index]


def describe_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(values),
        "min_ms": min(values),
        "p50_ms": median(values),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
    }


def build_index(
    dataset_root: Path,
    *,
    image_delta_good_ms: float,
    calib_delta_good_ms: float,
) -> dict[str, Any]:
    pcd_meta = load_json(dataset_root / "pcd" / "meta.json")
    calib_meta = load_json(dataset_root / "struct_json" / "meta.json")
    jpg_meta = load_json(dataset_root / "jpg" / "meta.json")

    pcd_entries = build_pcd_entries(dataset_root, pcd_meta)
    calib_stream = build_single_stream(
        dataset_root,
        calib_meta,
        file_key="calibration",
        base_dir="struct_json",
    )
    camera_streams = build_camera_streams(dataset_root, jpg_meta)
    active_cameras = available_cameras(camera_streams)
    skipped_cameras = {
        **SKIPPED_CAMERAS,
        **skipped_unavailable_cameras(camera_streams),
    }

    frames: list[dict[str, Any]] = []
    calib_deltas: list[float] = []
    image_deltas_by_camera: dict[str, list[float]] = {camera: [] for camera in active_cameras}
    invalid_frame_count = 0
    missing_detection_count = 0

    for pcd_entry in pcd_entries:
        pcd_ts = pcd_entry["timestamp"]
        calib_match, calib_delta_ns = nearest(calib_stream, pcd_ts)
        calib_delta = delta_ms(calib_delta_ns)
        calib_valid, calib_reason = match_quality(calib_delta, calib_delta_good_ms)
        if calib_delta is not None:
            calib_deltas.append(calib_delta)

        detection_exists = path_exists(dataset_root, pcd_entry["det_json"])
        if not detection_exists:
            missing_detection_count += 1

        frame_valid = detection_exists and calib_valid
        reasons: list[str] = []
        if not detection_exists:
            reasons.append("det_json_missing")
        if calib_reason:
            reasons.append(f"calib_{calib_reason}")

        images: dict[str, Any] = {}
        for camera_name in active_cameras:
            image_match, image_delta_ns = nearest(camera_streams[camera_name], pcd_ts)
            image_delta = delta_ms(image_delta_ns)
            image_valid, image_reason = match_quality(image_delta, image_delta_good_ms)
            image_exists = path_exists(dataset_root, image_match.path if image_match else None)

            if image_delta is not None:
                image_deltas_by_camera[camera_name].append(image_delta)
            if not image_exists:
                image_valid = False
                image_reason = "image_missing"

            if not image_valid:
                frame_valid = False
                reasons.append(f"{camera_name}_{image_reason}")

            images[camera_name] = {
                "timestamp": image_match.timestamp if image_match else None,
                "path": image_match.path if image_match else None,
                "delta_ms": image_delta,
                "calib_camera_name": camera_name,
                "valid_time_match": image_valid,
                "exists": image_exists,
                "reason": image_reason,
            }

        if not frame_valid:
            invalid_frame_count += 1

        frames.append(
            {
                "pcd_timestamp": pcd_ts,
                "pcd": {
                    "timestamp": pcd_ts,
                    "path": pcd_entry["pcd_path"],
                    "exists": path_exists(dataset_root, pcd_entry["pcd_path"]),
                },
                "det_json": pcd_entry["det_json"],
                "detection": {
                    "path": pcd_entry["det_json"],
                    "exists": detection_exists,
                },
                "calib": {
                    "timestamp": calib_match.timestamp if calib_match else None,
                    "path": calib_match.path if calib_match else None,
                    "delta_ms": calib_delta,
                    "valid_time_match": calib_valid,
                    "exists": path_exists(dataset_root, calib_match.path if calib_match else None),
                    "reason": calib_reason,
                },
                "images": images,
                "skipped_cameras": skipped_cameras,
                "valid_time_match": frame_valid,
                "reasons": reasons,
            }
        )

    per_camera_summary = {
        camera_name: {
            **describe_values(values),
            "invalid_count": sum(
                1 for frame in frames if not frame["images"][camera_name]["valid_time_match"]
            ),
            "stream_frame_count": len(camera_streams[camera_name]),
        }
        for camera_name, values in image_deltas_by_camera.items()
    }

    return {
        "schema_version": "projection_alignment_index.v1",
        "dataset_root": str(dataset_root),
        "primary_timeline": "pcd",
        "thresholds": {
            "image_delta_good_ms": image_delta_good_ms,
            "calib_delta_good_ms": calib_delta_good_ms,
        },
        "cameras": active_cameras,
        "skipped_cameras": skipped_cameras,
        "summary": {
            "pcd_frame_count": len(pcd_entries),
            "calib_frame_count": len(calib_stream),
            "camera_count": len(active_cameras),
            "invalid_frame_count": invalid_frame_count,
            "missing_detection_count": missing_detection_count,
            "calib_delta_ms": describe_values(calib_deltas),
            "image_delta_ms_by_camera": per_camera_summary,
        },
        "frames": frames,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build timestamp alignment index for projection visualization."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Dataset root directory. Default: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output aligned index JSON. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--image-delta-good-ms",
        type=float,
        default=50.0,
        help="Good nearest-neighbor image match threshold in milliseconds.",
    )
    parser.add_argument(
        "--calib-delta-good-ms",
        type=float,
        default=1000.0,
        help="Good nearest-neighbor calibration match threshold in milliseconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = build_index(
        args.dataset_root,
        image_delta_good_ms=args.image_delta_good_ms,
        calib_delta_good_ms=args.calib_delta_good_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = index["summary"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pcd_frame_count": summary["pcd_frame_count"],
                "calib_frame_count": summary["calib_frame_count"],
                "camera_count": summary["camera_count"],
                "invalid_frame_count": summary["invalid_frame_count"],
                "missing_detection_count": summary["missing_detection_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
