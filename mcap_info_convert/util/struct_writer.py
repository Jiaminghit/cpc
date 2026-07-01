"""Write SensorCalibration messages as JSON calibration files."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .mcap_reader import CalibrationMessage
except ImportError:  # Allow running files directly from this directory.
    from mcap_reader import CalibrationMessage


SENSOR_TYPE_NAMES = {
    1: "camera",
    2: "lidar",
    3: "radar",
    4: "ins",
}


@dataclass(frozen=True)
class WrittenCalibration:
    """One calibration JSON file written by this module."""

    timestamp_ns: int
    path: Path
    sensor_count: int


def _public_attribute_names(value: object) -> list[str]:
    names = []
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            attribute = getattr(value, name)
        except Exception:
            continue
        if callable(attribute):
            continue
        names.append(name)
    return sorted(names)


def to_jsonable(value: Any) -> Any:
    """Convert ROS dynamic objects into JSON-serializable Python values."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return list(value)
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    attributes = _public_attribute_names(value)
    if attributes:
        return {name: to_jsonable(getattr(value, name)) for name in attributes}
    return str(value)


def _sensor_type_name(sensor_type: int, sensor_name: str) -> str:
    if sensor_type in SENSOR_TYPE_NAMES:
        return SENSOR_TYPE_NAMES[sensor_type]

    lowered = sensor_name.lower()
    for candidate in ("camera", "lidar", "radar", "ins"):
        if candidate in lowered:
            return candidate
    return "unknown"


def _matrix_4x4(values: list[Any]) -> list[list[Any]] | None:
    if len(values) != 16:
        return None
    return [values[index : index + 4] for index in range(0, 16, 4)]


def extract_extrinsic(sensor: object) -> dict[str, Any]:
    """Extract common extrinsic fields from a SensorCalibItem."""

    extrinsics = getattr(sensor, "extrinsics", None)
    if extrinsics is None:
        return {}

    translation = to_jsonable(getattr(extrinsics, "translation", []))
    rotation = to_jsonable(getattr(extrinsics, "rotation", []))
    transform_matrix = to_jsonable(getattr(extrinsics, "transform_matrix", []))
    result = {
        "translation": translation,
        "rotation": rotation,
        "transform_matrix": transform_matrix,
    }
    matrix_4x4 = _matrix_4x4(transform_matrix)
    if matrix_4x4 is not None:
        result["transform_matrix_4x4"] = matrix_4x4
    return result


def extract_intrinsics(sensor: object) -> list[dict[str, Any]]:
    """Extract camera-like intrinsic fields from a SensorCalibItem."""

    intrinsics = getattr(sensor, "intrinsics", []) or []
    result = []
    for item in intrinsics:
        raw = to_jsonable(item)
        fx = raw.get("fx")
        fy = raw.get("fy")
        cx = raw.get("cx")
        cy = raw.get("cy")
        extracted = {
            "width": raw.get("width"),
            "height": raw.get("height"),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "camera_matrix": [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ]
            if None not in (fx, fy, cx, cy)
            else None,
            "distortion_model": raw.get("distortion_model"),
            "distortion_coeffs": raw.get("distortion_coeffs", []),
            "raw": raw,
        }
        result.append(extracted)
    return result


def extract_sensor(sensor: object) -> dict[str, Any]:
    """Convert a SensorCalibItem into a normalized sensor dictionary."""

    raw = to_jsonable(sensor)
    sensor_name = str(raw.get("sensor_name", ""))
    sensor_type_code = int(raw.get("sensor_type", 0) or 0)
    return {
        "name": sensor_name,
        "type": _sensor_type_name(sensor_type_code, sensor_name),
        "sensor_type_code": sensor_type_code,
        "sensor_sub_type": raw.get("sensor_sub_type"),
        "sensor_id": raw.get("sensor_id"),
        "serial_number": raw.get("serial_number"),
        "model": raw.get("model"),
        "calibration_state": raw.get("calibration_state"),
        "calibration_source": raw.get("calibration_source"),
        "has_intrinsics": bool(raw.get("has_intrinsics")),
        "intrinsics": extract_intrinsics(sensor),
        "extrinsic": extract_extrinsic(sensor),
        "raw": raw,
    }


def calibration_to_dict(message: CalibrationMessage) -> dict[str, Any]:
    """Convert one SensorCalibration message to a JSON-ready dictionary."""

    sensors = [extract_sensor(sensor) for sensor in message.sensors]
    return {
        "schema_version": "1.0",
        "timestamp": message.timestamp_ns,
        "log_time_ns": message.log_time_ns,
        "publish_time_ns": message.publish_time_ns,
        "source_mcap": message.source_file.name,
        "source_topic": message.topic,
        "frame_id": message.frame_id,
        "vehicle_model": message.vehicle_model,
        "vehicle_sub_model": message.vehicle_sub_model,
        "calibration_version": message.calibration_version,
        "sensor_count": len(sensors),
        "sensors": sensors,
        "raw": to_jsonable(message.ros_msg),
    }


def write_json(path: str | Path, value: dict[str, Any], *, overwrite: bool) -> Path:
    """Atomically write a JSON file."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {destination}; "
            "use --overwrite to replace existing JSON files"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temp_path, 0o644)
        os.replace(temp_path, destination)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return destination


def write_calibration_json(
    output_path: str | Path,
    message: CalibrationMessage,
    *,
    overwrite: bool = False,
) -> WrittenCalibration:
    """Write one SensorCalibration message to JSON."""

    value = calibration_to_dict(message)
    path = write_json(output_path, value, overwrite=overwrite)
    return WrittenCalibration(
        timestamp_ns=message.timestamp_ns,
        path=path,
        sensor_count=len(message.sensors),
    )
