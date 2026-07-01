"""Batch-convert SensorCalibration MCAP recordings to JSON files.

Example:
    python -m data_convert.util.mcap_to_struct \
        --input-dir /data/2067268107790897153/struct \
        --output-dir /data/2067268107790897153/struct_json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    from .mcap_reader import CalibrationMessage, inspect_mcap, iter_calibration_messages
    from .struct_writer import calibration_to_dict, write_calibration_json, write_json
except ImportError:  # Allow running this file directly from util/.
    from mcap_reader import CalibrationMessage, inspect_mcap, iter_calibration_messages
    from struct_writer import calibration_to_dict, write_calibration_json, write_json


logger = logging.getLogger("mcap_to_struct")


@dataclass(frozen=True)
class ConvertedCalibration:
    timestamp: int
    relative_path: str
    source_mcap: str
    source_topic: str
    sensor_count: int
    calibration_version: str


def discover_mcap_files(
    input_dir: str | Path | None,
    input_files: Iterable[str | Path],
) -> list[Path]:
    """Return a deterministic, duplicate-free MCAP input list."""

    candidates: list[Path] = []
    if input_dir is not None:
        directory = Path(input_dir)
        if not directory.is_dir():
            raise NotADirectoryError(f"input directory does not exist: {directory}")
        candidates.extend(sorted(directory.glob("*.mcap")))
    candidates.extend(Path(path) for path in input_files)

    files: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"MCAP file does not exist: {path}")
        if path.suffix.lower() != ".mcap":
            raise ValueError(f"input is not an .mcap file: {path}")
        seen.add(path)
        files.append(path)

    if not files:
        raise FileNotFoundError("no .mcap input files found")
    return files


def _format_timestamp(timestamp_ns: int | None) -> str:
    if timestamp_ns is None:
        return "-"
    return str(timestamp_ns)


def print_inspection(paths: Iterable[Path]) -> None:
    """Print topic/schema/message-count information for each MCAP."""

    for path in paths:
        info = inspect_mcap(path)
        print(f"\nMCAP: {info.path}")
        print(
            f"  messages={info.total_message_count} "
            f"start_ns={_format_timestamp(info.message_start_time_ns)} "
            f"end_ns={_format_timestamp(info.message_end_time_ns)}"
        )
        for topic in info.topics:
            print(
                f"  {topic.topic}: count={topic.message_count}, "
                f"schema={topic.schema_name}, encoding={topic.message_encoding}"
            )


def _unique_destination(
    calib_dir: Path,
    timestamp_ns: int,
    used_timestamps: set[int],
    overwrite: bool,
) -> Path:
    if timestamp_ns in used_timestamps:
        raise ValueError(f"duplicate calibration timestamp in input: {timestamp_ns}")
    used_timestamps.add(timestamp_ns)

    destination = calib_dir / f"{timestamp_ns}.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {destination}; "
            "use --overwrite to replace existing calibration JSON files"
        )
    return destination


def convert_mcaps(
    paths: Iterable[Path],
    output_dir: str | Path,
    *,
    topic: str = "/calib/calib_param",
    max_frames: int | None = None,
    overwrite: bool = False,
) -> tuple[list[ConvertedCalibration], CalibrationMessage]:
    """Convert selected SensorCalibration messages from multiple MCAP files."""

    root = Path(output_dir).resolve()
    calib_dir = root / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    converted: list[ConvertedCalibration] = []
    used_timestamps: set[int] = set()
    latest_message: CalibrationMessage | None = None

    for mcap_path in paths:
        file_count = 0
        logger.info("Reading %s (topic=%s)", mcap_path, topic)
        for message in iter_calibration_messages(mcap_path, topic=topic):
            if max_frames is not None and len(converted) >= max_frames:
                break

            destination = _unique_destination(
                calib_dir,
                message.timestamp_ns,
                used_timestamps,
                overwrite,
            )
            written = write_calibration_json(
                destination,
                message,
                overwrite=overwrite,
            )
            converted.append(
                ConvertedCalibration(
                    timestamp=message.timestamp_ns,
                    relative_path=written.path.relative_to(root).as_posix(),
                    source_mcap=mcap_path.name,
                    source_topic=topic,
                    sensor_count=written.sensor_count,
                    calibration_version=message.calibration_version,
                )
            )
            latest_message = message
            file_count += 1
            logger.info(
                "Converted calibration %d: %s (%d sensors)",
                len(converted),
                destination,
                written.sensor_count,
            )

        if file_count == 0 and not (
            max_frames is not None and len(converted) >= max_frames
        ):
            raise ValueError(f"no messages found on topic {topic!r} in {mcap_path}")
        if max_frames is not None and len(converted) >= max_frames:
            break

    if latest_message is None:
        raise RuntimeError(f"no calibration messages converted from topic {topic!r}")

    latest_path = root / "calibration_latest.json"
    write_json(
        latest_path,
        calibration_to_dict(latest_message),
        overwrite=overwrite,
    )
    return sorted(converted, key=lambda item: item.timestamp), latest_message


def _sensor_summary(message: CalibrationMessage) -> list[dict]:
    sensors = []
    for sensor in message.sensors:
        sensor_name = str(getattr(sensor, "sensor_name", ""))
        sensor_type = int(getattr(sensor, "sensor_type", 0) or 0)
        sensors.append(
            {
                "name": sensor_name,
                "type": {
                    1: "camera",
                    2: "lidar",
                    3: "radar",
                    4: "ins",
                }.get(sensor_type, "unknown"),
                "sensor_type_code": sensor_type,
                "sensor_id": int(getattr(sensor, "sensor_id", 0) or 0),
                "has_intrinsics": bool(getattr(sensor, "has_intrinsics", 0)),
            }
        )
    return sensors


def build_meta(
    dataset_name: str,
    topic: str,
    frames: Iterable[ConvertedCalibration],
    latest_message: CalibrationMessage,
) -> dict:
    """Build the struct conversion metadata file."""

    ordered = sorted(frames, key=lambda item: item.timestamp)
    latest = ordered[-1]
    return {
        "schema_version": "1.0",
        "dataset_type_code": "struct",
        "dataset_name": dataset_name,
        "frame_count": len(ordered),
        "has_preannotation": False,
        "preannotation_models": [],
        "sensors": _sensor_summary(latest_message),
        "frame_map": [
            {
                "timestamp": frame.timestamp,
                "files": {"calibration": frame.relative_path},
                "preann": {},
            }
            for frame in ordered
        ],
        "metadata": {
            "conversion": {
                "source_format": "mcap",
                "target_format": "json",
                "topic": topic,
                "latest_timestamp": latest.timestamp,
                "latest_file": "calibration_latest.json",
                "frames": [asdict(frame) for frame in ordered],
            }
        },
    }


def _atomic_write_json(path: Path, value: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"metadata already exists: {path}; use --overwrite to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temp_path, 0o644)
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert ROS2 SensorCalibration messages in MCAP files to JSON"
    )
    parser.add_argument(
        "--input-dir",
        help="Directory containing .mcap files (non-recursive)",
    )
    parser.add_argument(
        "--input-file",
        action="append",
        default=[],
        help="Individual .mcap file; may be specified multiple times",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Dataset output root; JSON files are written under "
            "<output>/calib/"
        ),
    )
    parser.add_argument("--topic", default="/calib/calib_param")
    parser.add_argument(
        "--dataset-name",
        help="Dataset name written to meta.json; defaults to output directory name",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Stop after this many calibration messages (useful for validation)",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print MCAP topics and message counts without converting",
    )
    parser.add_argument(
        "--no-meta",
        action="store_true",
        help="Do not generate meta.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing JSON and meta.json files",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.input_dir and not args.input_file:
        parser.error("one of --input-dir or --input-file is required")
    if not args.inspect_only and not args.output_dir:
        parser.error("--output-dir is required unless --inspect-only is used")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be greater than zero")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    paths = discover_mcap_files(args.input_dir, args.input_file)
    if args.inspect_only:
        print_inspection(paths)
        return

    output_dir = Path(args.output_dir).resolve()
    frames, latest_message = convert_mcaps(
        paths,
        output_dir,
        topic=args.topic,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
    )
    if not frames:
        raise RuntimeError(f"no calibration messages converted from {args.topic!r}")

    meta_path = None
    if not args.no_meta:
        meta = build_meta(
            dataset_name=args.dataset_name or output_dir.name,
            topic=args.topic,
            frames=frames,
            latest_message=latest_message,
        )
        meta_path = output_dir / "meta.json"
        _atomic_write_json(meta_path, meta, overwrite=args.overwrite)

    logger.info(
        "Done: converted=%d output=%s latest=%s meta=%s",
        len(frames),
        output_dir,
        output_dir / "calibration_latest.json",
        meta_path or "(disabled)",
    )


if __name__ == "__main__":
    main()
