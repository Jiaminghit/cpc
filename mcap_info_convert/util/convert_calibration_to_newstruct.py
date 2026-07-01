#!/usr/bin/env python3
"""Convert calibration JSON files to the compact newstruct schema.

The converter rewrites:
  * calibration_latest.json
  * calib/*.json

It preserves each frame's calibration values, while dropping wrapper metadata
such as schema_version, timestamp, raw, and transform_matrix_4x4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_TOP_KEYS = ["calibration_version", "vehicle_model", "vehicle_sub_model", "sensors"]
EXPECTED_SENSOR_KEYS = [
    "sensor_name",
    "sensor_type",
    "sensor_id",
    "is_valid",
    "calibration_state",
    "calibration_source",
    "update",
    "extrinsics",
    "has_intrinsics",
    "intrinsics",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def int_flag(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    return value


def ordered_update(update: dict[str, Any] | None) -> dict[str, Any]:
    update = update or {}
    return {
        "sec": update.get("sec", 0),
        "nanosec": update.get("nanosec", 0),
    }


def ordered_extrinsics(extrinsics: dict[str, Any] | None) -> dict[str, Any]:
    extrinsics = extrinsics or {}
    return {
        "translation": extrinsics.get("translation", []),
        "rotation": extrinsics.get("rotation", []),
        "transform_matrix": extrinsics.get("transform_matrix", []),
    }


def convert_sensor(sensor: dict[str, Any]) -> dict[str, Any]:
    raw = sensor.get("raw") or sensor
    extrinsics = raw.get("extrinsics") or sensor.get("extrinsic") or sensor.get("extrinsics")
    update = raw.get("update") or sensor.get("update")

    return {
        "sensor_name": raw.get("sensor_name", sensor.get("name", sensor.get("sensor_name", ""))),
        "sensor_type": raw.get("sensor_type", sensor.get("sensor_type_code", sensor.get("sensor_type", 0))),
        "sensor_id": raw.get("sensor_id", sensor.get("sensor_id", 0)),
        "is_valid": raw.get("is_valid", sensor.get("is_valid", True)),
        "calibration_state": raw.get(
            "calibration_state",
            sensor.get("calibration_state", 0),
        ),
        "calibration_source": raw.get(
            "calibration_source",
            sensor.get("calibration_source", 0),
        ),
        "update": ordered_update(update),
        "extrinsics": ordered_extrinsics(extrinsics),
        "has_intrinsics": int_flag(
            raw.get("has_intrinsics", int(bool(sensor.get("has_intrinsics", False))))
        ),
        "intrinsics": raw.get("intrinsics", sensor.get("intrinsics", [])),
    }


def convert_calibration(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "calibration_version": data.get("calibration_version", ""),
        "vehicle_model": data.get("vehicle_model", ""),
        "vehicle_sub_model": data.get("vehicle_sub_model", ""),
        "sensors": [convert_sensor(sensor) for sensor in data.get("sensors", [])],
    }


def calibration_files(struct_json_dir: Path) -> list[Path]:
    files: list[Path] = []
    latest = struct_json_dir / "calibration_latest.json"
    if latest.is_file():
        files.append(latest)
    files.extend(sorted((struct_json_dir / "calib").glob("*.json")))
    if not files:
        raise FileNotFoundError(f"no calibration JSON files found under {struct_json_dir}")
    return files


def validate_converted(path: Path, data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if list(data.keys()) != EXPECTED_TOP_KEYS:
        errors.append(f"{path}: top keys are {list(data.keys())}")
    sensors = data.get("sensors", [])
    if not isinstance(sensors, list):
        errors.append(f"{path}: sensors is not a list")
        return errors
    for index, sensor in enumerate(sensors):
        if list(sensor.keys()) != EXPECTED_SENSOR_KEYS:
            errors.append(f"{path}: sensor {index} keys are {list(sensor.keys())}")
        if list(sensor.get("update", {}).keys()) != ["sec", "nanosec"]:
            errors.append(f"{path}: sensor {index} update keys are {list(sensor.get('update', {}).keys())}")
        if list(sensor.get("extrinsics", {}).keys()) != ["translation", "rotation", "transform_matrix"]:
            errors.append(
                f"{path}: sensor {index} extrinsics keys are {list(sensor.get('extrinsics', {}).keys())}"
            )
    return errors


def rewrite_files(files: list[Path], *, dry_run: bool) -> tuple[int, list[str]]:
    errors: list[str] = []
    for path in files:
        converted = convert_calibration(load_json(path))
        errors.extend(validate_converted(path, converted))
        if not dry_run:
            path.write_text(
                json.dumps(converted, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    return len(files), errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert struct_json calibration files to the compact newstruct schema."
    )
    parser.add_argument(
        "struct_json_dir",
        type=Path,
        help="Path to a dataset's struct_json directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate conversion without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = calibration_files(args.struct_json_dir)
    count, errors = rewrite_files(files, dry_run=args.dry_run)
    result = {
        "struct_json_dir": str(args.struct_json_dir),
        "file_count": count,
        "dry_run": args.dry_run,
        "error_count": len(errors),
        "errors": errors[:20],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
