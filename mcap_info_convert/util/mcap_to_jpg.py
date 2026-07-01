"""Batch-convert camera MCAP recordings to JPG image datasets.

Example:
    python -m data_convert.util.mcap_to_jpg \
        --input-dir /data/2067268107790897153/camera \
        --output-dir /data/2067268107790897153/jpg

    python -m data_convert.util.mcap_to_jpg \
        --input-file /data/camera/video_001.mcap \
        --output-dir /data/camera_jpg \
        --topic front_long=/sensor/camera_front_long_image
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    from .jpg_writer import WrittenJpg, write_jpg_sequence
    from .mcap_reader import ImageFrame, inspect_mcap, iter_image_frames
except ImportError:  # Allow running this file directly from util/.
    from jpg_writer import WrittenJpg, write_jpg_sequence
    from mcap_reader import ImageFrame, inspect_mcap, iter_image_frames


logger = logging.getLogger("mcap_to_jpg")


@dataclass(frozen=True)
class TopicSpec:
    sensor_name: str
    topic: str


@dataclass(frozen=True)
class ConvertedImage:
    timestamp: int
    sensor_name: str
    relative_path: str
    source_mcap: str
    source_topic: str
    width: int
    height: int
    source_encoding: str
    source_message_index: int | None


@dataclass(frozen=True)
class SensorConversionStats:
    sensor_name: str
    source_topic: str
    input_messages: int
    decoded_frames: int


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


def sensor_name_from_topic(topic: str) -> str:
    """Build a stable folder-friendly sensor name from a camera topic."""

    name = topic.strip().strip("/")
    if name.startswith("sensor/"):
        name = name[len("sensor/") :]
    if name.endswith("_image"):
        name = name[: -len("_image")]
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = name.strip("_")
    if not name:
        raise ValueError(f"cannot derive sensor name from topic: {topic!r}")
    return name


def parse_topic_specs(raw_topics: Iterable[str]) -> list[TopicSpec]:
    """Parse --topic values.

    Accepted forms:
      * /sensor/camera_front_long_image
      * camera_front_long=/sensor/camera_front_long_image
    """

    specs: list[TopicSpec] = []
    seen_sensors: set[str] = set()
    seen_topics: set[str] = set()
    for raw in raw_topics:
        value = raw.strip()
        if not value:
            raise ValueError("--topic cannot be empty")
        if "=" in value:
            sensor_name, topic = value.split("=", 1)
            sensor_name = sensor_name.strip()
            topic = topic.strip()
        else:
            topic = value
            sensor_name = sensor_name_from_topic(topic)
        if not topic.startswith("/"):
            raise ValueError(f"topic must start with '/': {topic!r}")
        if not sensor_name:
            raise ValueError(f"sensor name cannot be empty for topic: {topic!r}")
        if sensor_name in seen_sensors:
            raise ValueError(f"duplicate sensor name: {sensor_name}")
        if topic in seen_topics:
            raise ValueError(f"duplicate topic: {topic}")
        seen_sensors.add(sensor_name)
        seen_topics.add(topic)
        specs.append(TopicSpec(sensor_name=sensor_name, topic=topic))
    return specs


def discover_image_topic_specs(paths: Iterable[Path]) -> list[TopicSpec]:
    """Discover ImageV2 topics from MCAP summaries."""

    topics: dict[str, str] = {}
    for path in paths:
        info = inspect_mcap(path)
        for topic in info.topics:
            if topic.schema_name.endswith("ImageV2"):
                topics.setdefault(topic.topic, sensor_name_from_topic(topic.topic))

    specs = [
        TopicSpec(sensor_name=sensor_name, topic=topic)
        for topic, sensor_name in sorted(topics.items())
    ]
    if not specs:
        raise ValueError("no ImageV2 topics found; pass --topic explicitly")
    return specs


def _collect_frames(
    mcap_path: Path,
    topic: str,
    remaining_messages: int | None,
) -> list[ImageFrame]:
    frames: list[ImageFrame] = []
    for frame in iter_image_frames(mcap_path, topic=topic):
        if remaining_messages is not None and len(frames) >= remaining_messages:
            break
        frames.append(frame)
    return frames


def _converted_records(
    *,
    written: Iterable[WrittenJpg],
    root: Path,
    sensor_name: str,
    source_mcap: str,
    source_topic: str,
) -> list[ConvertedImage]:
    records: list[ConvertedImage] = []
    for item in written:
        records.append(
            ConvertedImage(
                timestamp=item.timestamp_ns,
                sensor_name=sensor_name,
                relative_path=item.path.relative_to(root).as_posix(),
                source_mcap=source_mcap,
                source_topic=source_topic,
                width=item.width,
                height=item.height,
                source_encoding=item.source_encoding,
                source_message_index=item.source_message_index,
            )
        )
    return records


def convert_mcaps(
    paths: Iterable[Path],
    output_dir: str | Path,
    *,
    topic_specs: Iterable[TopicSpec],
    max_frames: int | None = None,
    overwrite: bool = False,
    quality: int = 95,
    show_decoder_log: bool = False,
) -> tuple[list[ConvertedImage], list[SensorConversionStats]]:
    """Convert selected camera topics from multiple MCAP files."""

    root = Path(output_dir).resolve()
    converted: list[ConvertedImage] = []
    stats_by_sensor: dict[str, SensorConversionStats] = {}
    used_timestamps: dict[str, set[int]] = {}
    input_counts: dict[str, int] = {}
    decoded_counts: dict[str, int] = {}

    specs = list(topic_specs)
    for spec in specs:
        used_timestamps[spec.sensor_name] = set()
        input_counts[spec.sensor_name] = 0
        decoded_counts[spec.sensor_name] = 0

    for spec in specs:
        sensor_dir = root / spec.sensor_name
        sensor_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Converting topic %s -> sensor %s", spec.topic, spec.sensor_name)

        for mcap_path in paths:
            if max_frames is not None and input_counts[spec.sensor_name] >= max_frames:
                break
            remaining = (
                max_frames - input_counts[spec.sensor_name]
                if max_frames is not None
                else None
            )
            frames = _collect_frames(mcap_path, spec.topic, remaining)
            if not frames:
                continue

            input_counts[spec.sensor_name] += len(frames)
            written = write_jpg_sequence(
                sensor_dir,
                frames,
                overwrite=overwrite,
                quality=quality,
                used_timestamps=used_timestamps[spec.sensor_name],
                show_decoder_log=show_decoder_log,
            )
            decoded_counts[spec.sensor_name] += len(written)
            converted.extend(
                _converted_records(
                    written=written,
                    root=root,
                    sensor_name=spec.sensor_name,
                    source_mcap=mcap_path.name,
                    source_topic=spec.topic,
                )
            )
            logger.info(
                "Converted %s from %s: input_messages=%d decoded_jpg=%d",
                spec.sensor_name,
                mcap_path.name,
                len(frames),
                len(written),
            )

        stats_by_sensor[spec.sensor_name] = SensorConversionStats(
            sensor_name=spec.sensor_name,
            source_topic=spec.topic,
            input_messages=input_counts[spec.sensor_name],
            decoded_frames=decoded_counts[spec.sensor_name],
        )
        if input_counts[spec.sensor_name] == 0:
            raise ValueError(f"no messages found on topic {spec.topic!r}")
        if decoded_counts[spec.sensor_name] == 0:
            raise RuntimeError(f"no JPG frames decoded for topic {spec.topic!r}")

    return (
        sorted(converted, key=lambda item: (item.timestamp, item.sensor_name)),
        [stats_by_sensor[spec.sensor_name] for spec in specs],
    )


def build_meta(
    dataset_name: str,
    topic_specs: Iterable[TopicSpec],
    frames: Iterable[ConvertedImage],
    stats: Iterable[SensorConversionStats],
) -> dict:
    """Build the camera conversion metadata file."""

    ordered_frames = sorted(frames, key=lambda item: (item.timestamp, item.sensor_name))
    frame_map: dict[int, dict[str, str]] = {}
    for frame in ordered_frames:
        frame_map.setdefault(frame.timestamp, {})[frame.sensor_name] = frame.relative_path

    stats_by_sensor = {item.sensor_name: item for item in stats}
    sensors = []
    for spec in topic_specs:
        sensor_stats = stats_by_sensor[spec.sensor_name]
        sensors.append(
            {
                "name": spec.sensor_name,
                "type": "camera",
                "source_topic": spec.topic,
                "input_messages": sensor_stats.input_messages,
                "decoded_frames": sensor_stats.decoded_frames,
            }
        )

    return {
        "schema_version": "1.0",
        "dataset_type_code": "camera",
        "dataset_name": dataset_name,
        "frame_count": len(frame_map),
        "image_count": len(ordered_frames),
        "has_preannotation": False,
        "preannotation_models": [],
        "sensors": sensors,
        "frame_map": [
            {
                "timestamp": timestamp,
                "files": files,
                "preann": {},
            }
            for timestamp, files in sorted(frame_map.items())
        ],
        "metadata": {
            "conversion": {
                "source_format": "mcap",
                "target_format": "jpg",
                "topics": [asdict(spec) for spec in topic_specs],
                "sensor_stats": [asdict(item) for item in stats],
                "frames": [asdict(frame) for frame in ordered_frames],
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
        description="Convert ROS2 ImageV2 messages in MCAP files to JPG"
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
        help="Dataset output root; JPGs are written under <output>/<sensor>/",
    )
    parser.add_argument(
        "--topic",
        action="append",
        default=[],
        help=(
            "Camera topic to convert. Use /topic or sensor_name=/topic. "
            "May be specified multiple times. If omitted, ImageV2 topics "
            "are discovered from the MCAP summary."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        help="Dataset name written to meta.json; defaults to output directory name",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Stop after this many input messages per sensor (useful for validation)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100",
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
        help="Replace existing JPG and meta.json files",
    )
    parser.add_argument(
        "--show-decoder-log",
        action="store_true",
        help="Show native FFmpeg/OpenCV decoder warnings while reading H265 streams",
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
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    paths = discover_mcap_files(args.input_dir, args.input_file)
    if args.inspect_only:
        print_inspection(paths)
        return

    topic_specs = (
        parse_topic_specs(args.topic)
        if args.topic
        else discover_image_topic_specs(paths)
    )
    output_dir = Path(args.output_dir).resolve()
    frames, stats = convert_mcaps(
        paths,
        output_dir,
        topic_specs=topic_specs,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
        quality=args.quality,
        show_decoder_log=args.show_decoder_log,
    )
    if not frames:
        raise RuntimeError("no JPG frames converted")

    meta_path = None
    if not args.no_meta:
        meta = build_meta(
            dataset_name=args.dataset_name or output_dir.name,
            topic_specs=topic_specs,
            frames=frames,
            stats=stats,
        )
        meta_path = output_dir / "meta.json"
        _atomic_write_json(meta_path, meta, overwrite=args.overwrite)

    logger.info(
        "Done: images=%d sensors=%d output=%s meta=%s",
        len(frames),
        len(topic_specs),
        output_dir,
        meta_path or "(disabled)",
    )


if __name__ == "__main__":
    main()
