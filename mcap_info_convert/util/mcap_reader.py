"""Read typed ROS2 messages from MCAP recordings.

This module has two responsibilities:

* common MCAP/ROS2 utilities such as dependency loading, topic inspection,
  generic message iteration and timestamp extraction;
* typed adapters for the message families used by the converters, including
  lidar PointCloud2V2, camera ImageV2 and calibration SensorCalibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


POINT_FIELD_TYPES: dict[int, tuple[str, int, str]] = {
    1: ("int8", 1, "I"),
    2: ("uint8", 1, "U"),
    3: ("int16", 2, "I"),
    4: ("uint16", 2, "U"),
    5: ("int32", 4, "I"),
    6: ("uint32", 4, "U"),
    7: ("float32", 4, "F"),
    8: ("float64", 8, "F"),
}


@dataclass(frozen=True)
class PointField:
    """Description of one field inside a point record."""

    name: str
    offset: int
    datatype: int
    count: int

    @property
    def size(self) -> int:
        try:
            _, item_size, _ = POINT_FIELD_TYPES[self.datatype]
        except KeyError as exc:
            raise ValueError(
                f"unsupported PointField datatype {self.datatype} for {self.name!r}"
            ) from exc
        return item_size * self.count

    @property
    def pcd_size(self) -> int:
        return POINT_FIELD_TYPES[self.datatype][1]

    @property
    def pcd_type(self) -> str:
        return POINT_FIELD_TYPES[self.datatype][2]


@dataclass(frozen=True)
class PointCloudFrame:
    """One decoded PointCloud2V2 message without copying its point bytes."""

    timestamp_ns: int
    log_time_ns: int
    publish_time_ns: int
    topic: str
    frame_id: str
    width: int
    height: int
    fields: tuple[PointField, ...]
    is_bigendian: bool
    point_step: int
    row_step: int
    data: bytes
    source_file: Path

    @property
    def point_count(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class ImageFrame:
    """One decoded ImageV2 message without interpreting its pixel encoding."""

    timestamp_ns: int
    log_time_ns: int
    publish_time_ns: int
    topic: str
    frame_id: str
    width: int
    height: int
    encoding: str
    is_bigendian: bool
    step: int
    data: bytes
    source_file: Path
    stamp_start_ns: int | None = None
    stamp_end_ns: int | None = None


@dataclass(frozen=True)
class CalibrationMessage:
    """One decoded SensorCalibration message plus its MCAP timing metadata."""

    timestamp_ns: int
    log_time_ns: int
    publish_time_ns: int
    topic: str
    frame_id: str
    vehicle_model: str
    vehicle_sub_model: str
    calibration_version: str
    sensors: tuple[object, ...]
    ros_msg: object
    source_file: Path


@dataclass(frozen=True)
class TopicInfo:
    """MCAP channel information used by the inspect command."""

    topic: str
    schema_name: str
    message_encoding: str
    message_count: int


@dataclass(frozen=True)
class McapInfo:
    """Summary of one MCAP file."""

    path: Path
    message_start_time_ns: int | None
    message_end_time_ns: int | None
    total_message_count: int
    topics: tuple[TopicInfo, ...]


@dataclass(frozen=True)
class RosMessageEnvelope:
    """Generic ROS2 message read from MCAP with normalized metadata."""

    timestamp_ns: int
    log_time_ns: int
    publish_time_ns: int
    topic: str
    schema_name: str
    message_encoding: str
    ros_msg: object
    source_file: Path


def _require_mcap():
    try:
        from mcap.reader import make_reader
        from mcap_ros2.reader import read_ros2_messages
    except ImportError as exc:
        raise RuntimeError(
            "MCAP dependencies are missing. Install requirements-dev.txt "
            "(mcap and mcap-ros2-support)."
        ) from exc
    return make_reader, read_ros2_messages


def inspect_mcap(path: str | Path) -> McapInfo:
    """Read channel and time-range metadata without decoding point messages."""

    make_reader, _ = _require_mcap()
    mcap_path = Path(path)
    with mcap_path.open("rb") as stream:
        summary = make_reader(stream).get_summary()

    if summary is None:
        raise ValueError(f"MCAP file has no summary section: {mcap_path}")

    statistics = summary.statistics
    channel_counts = statistics.channel_message_counts if statistics else {}
    topics = []
    for channel_id, channel in sorted(
        summary.channels.items(), key=lambda item: item[1].topic
    ):
        schema = summary.schemas.get(channel.schema_id)
        topics.append(
            TopicInfo(
                topic=channel.topic,
                schema_name=schema.name if schema else "",
                message_encoding=channel.message_encoding,
                message_count=channel_counts.get(channel_id, 0),
            )
        )

    return McapInfo(
        path=mcap_path,
        message_start_time_ns=statistics.message_start_time if statistics else None,
        message_end_time_ns=statistics.message_end_time if statistics else None,
        total_message_count=statistics.message_count if statistics else 0,
        topics=tuple(topics),
    )


def _stamp_to_ns(stamp: object | None) -> int | None:
    if stamp is None:
        return None
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    timestamp_ns = sec * 1_000_000_000 + nanosec
    return timestamp_ns if timestamp_ns > 0 else None


def _message_timestamp_ns(ros_msg, publish_time_ns: int, log_time_ns: int) -> int:
    """Prefer the sensor acquisition timestamp from the ROS message header."""

    header = getattr(ros_msg, "header", None)
    timestamp_ns = _stamp_to_ns(getattr(header, "stamp", None))
    if timestamp_ns is not None:
        return timestamp_ns
    return publish_time_ns or log_time_ns


def _require_attributes(
    ros_msg: object,
    names: Sequence[str],
    *,
    path: Path,
    topic: str,
    expected_type: str,
) -> None:
    missing = [name for name in names if not hasattr(ros_msg, name)]
    if missing:
        raise TypeError(
            f"{path}: topic {topic!r} is not {expected_type}; "
            f"missing attributes: {missing}"
        )


def iter_ros2_messages(
    path: str | Path,
    topics: Sequence[str] | None = None,
) -> Iterator[RosMessageEnvelope]:
    """Yield decoded ROS2 messages with common MCAP metadata.

    Args:
        path: MCAP file path.
        topics: Optional topic filter.  Pass ``None`` to iterate all topics.
    """

    _, read_ros2_messages = _require_mcap()
    mcap_path = Path(path)
    kwargs = {"topics": list(topics)} if topics is not None else {}
    for message in read_ros2_messages(mcap_path, **kwargs):
        log_time_ns = int(message.log_time_ns)
        publish_time_ns = int(message.publish_time_ns)
        ros_msg = message.ros_msg
        yield RosMessageEnvelope(
            timestamp_ns=_message_timestamp_ns(
                ros_msg,
                publish_time_ns,
                log_time_ns,
            ),
            log_time_ns=log_time_ns,
            publish_time_ns=publish_time_ns,
            topic=message.channel.topic,
            schema_name=message.schema.name if message.schema else "",
            message_encoding=message.channel.message_encoding,
            ros_msg=ros_msg,
            source_file=mcap_path,
        )


def _convert_fields(raw_fields: Sequence[object]) -> tuple[PointField, ...]:
    fields = tuple(
        PointField(
            name=str(field.name),
            offset=int(field.offset),
            datatype=int(field.datatype),
            count=int(field.count),
        )
        for field in raw_fields
    )
    if not fields:
        raise ValueError("PointCloud2V2 message contains no point fields")

    names = [field.name for field in fields]
    if len(names) != len(set(names)):
        raise ValueError(f"PointCloud2V2 contains duplicate field names: {names}")

    # Accessing size validates every datatype before conversion starts.
    for field in fields:
        if field.count <= 0:
            raise ValueError(f"invalid count for point field {field.name!r}")
        _ = field.size
    return fields


def iter_pointcloud_frames(
    path: str | Path,
    topic: str = "/lidar/pandar",
) -> Iterator[PointCloudFrame]:
    """Yield point-cloud frames from one MCAP file in timestamp order."""

    for envelope in iter_ros2_messages(path, topics=[topic]):
        ros_msg = envelope.ros_msg
        required_attributes = (
            "width",
            "height",
            "fields",
            "point_step",
            "row_step",
            "data",
        )
        _require_attributes(
            ros_msg,
            required_attributes,
            path=envelope.source_file,
            topic=topic,
            expected_type="PointCloud2-like",
        )

        width = int(ros_msg.width)
        height = int(ros_msg.height)
        point_step = int(ros_msg.point_step)
        row_step = int(ros_msg.row_step)
        data = bytes(ros_msg.data)
        expected_size = row_step * height
        if len(data) < expected_size:
            raise ValueError(
                f"{envelope.source_file}: truncated point data on {topic}: "
                f"got {len(data)} bytes, expected at least {expected_size}"
            )
        if point_step <= 0 or row_step < width * point_step:
            raise ValueError(
                f"{envelope.source_file}: invalid point layout: width={width}, "
                f"point_step={point_step}, row_step={row_step}"
            )

        fields = _convert_fields(ros_msg.fields)
        for field in fields:
            if field.offset < 0 or field.offset + field.size > point_step:
                raise ValueError(
                    f"{envelope.source_file}: field {field.name!r} exceeds point_step "
                    f"({field.offset}+{field.size}>{point_step})"
                )

        header = getattr(ros_msg, "header", None)
        yield PointCloudFrame(
            timestamp_ns=envelope.timestamp_ns,
            log_time_ns=envelope.log_time_ns,
            publish_time_ns=envelope.publish_time_ns,
            topic=envelope.topic,
            frame_id=str(getattr(header, "frame_id", "")),
            width=width,
            height=height,
            fields=fields,
            is_bigendian=bool(getattr(ros_msg, "is_bigendian", False)),
            point_step=point_step,
            row_step=row_step,
            data=data,
            source_file=envelope.source_file,
        )


def iter_image_frames(
    path: str | Path,
    topic: str,
) -> Iterator[ImageFrame]:
    """Yield camera ImageV2 frames from one MCAP file."""

    for envelope in iter_ros2_messages(path, topics=[topic]):
        ros_msg = envelope.ros_msg
        required_attributes = (
            "width",
            "height",
            "encoding",
            "step",
            "data",
        )
        _require_attributes(
            ros_msg,
            required_attributes,
            path=envelope.source_file,
            topic=topic,
            expected_type="Image-like",
        )

        width = int(ros_msg.width)
        height = int(ros_msg.height)
        step = int(ros_msg.step)
        data = bytes(ros_msg.data)
        if width <= 0 or height <= 0:
            raise ValueError(
                f"{envelope.source_file}: invalid image layout on {topic}: "
                f"width={width}, height={height}, step={step}"
            )
        if not data:
            raise ValueError(
                f"{envelope.source_file}: empty image data on {topic}"
            )
        if step > 0:
            expected_size = step * height
            if len(data) < expected_size:
                raise ValueError(
                    f"{envelope.source_file}: truncated image data on {topic}: "
                    f"got {len(data)} bytes, expected at least {expected_size}"
                )

        header = getattr(ros_msg, "header", None)
        yield ImageFrame(
            timestamp_ns=envelope.timestamp_ns,
            log_time_ns=envelope.log_time_ns,
            publish_time_ns=envelope.publish_time_ns,
            topic=envelope.topic,
            frame_id=str(getattr(header, "frame_id", "")),
            width=width,
            height=height,
            encoding=str(ros_msg.encoding),
            is_bigendian=bool(getattr(ros_msg, "is_bigendian", False)),
            step=step,
            data=data,
            source_file=envelope.source_file,
            stamp_start_ns=_stamp_to_ns(getattr(ros_msg, "stamp_start", None)),
            stamp_end_ns=_stamp_to_ns(getattr(ros_msg, "stamp_end", None)),
        )


def iter_calibration_messages(
    path: str | Path,
    topic: str = "/calib/calib_param",
) -> Iterator[CalibrationMessage]:
    """Yield SensorCalibration messages from one MCAP file."""

    for envelope in iter_ros2_messages(path, topics=[topic]):
        ros_msg = envelope.ros_msg
        required_attributes = (
            "calibration_version",
            "sensors",
            "vehicle_model",
            "vehicle_sub_model",
        )
        _require_attributes(
            ros_msg,
            required_attributes,
            path=envelope.source_file,
            topic=topic,
            expected_type="SensorCalibration-like",
        )

        header = getattr(ros_msg, "header", None)
        yield CalibrationMessage(
            timestamp_ns=envelope.timestamp_ns,
            log_time_ns=envelope.log_time_ns,
            publish_time_ns=envelope.publish_time_ns,
            topic=envelope.topic,
            frame_id=str(getattr(header, "frame_id", "")),
            vehicle_model=str(ros_msg.vehicle_model),
            vehicle_sub_model=str(ros_msg.vehicle_sub_model),
            calibration_version=str(ros_msg.calibration_version),
            sensors=tuple(ros_msg.sensors),
            ros_msg=ros_msg,
            source_file=envelope.source_file,
        )
