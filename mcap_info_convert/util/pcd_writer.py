"""Write PointCloud2-style binary point records as PCD files."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import BinaryIO

try:
    from .mcap_reader import PointCloudFrame, PointField
except ImportError:  # Allow running files directly from this directory.
    from mcap_reader import PointCloudFrame, PointField


def _ordered_fields(frame: PointCloudFrame) -> tuple[PointField, ...]:
    fields = tuple(sorted(frame.fields, key=lambda field: field.offset))
    if not {"x", "y", "z"}.issubset(field.name for field in fields):
        raise ValueError(
            f"point cloud must contain x/y/z fields; "
            f"got {[field.name for field in fields]}"
        )
    return fields


def _pcd_header(frame: PointCloudFrame, fields: tuple[PointField, ...]) -> bytes:
    field_names = " ".join(field.name for field in fields)
    sizes = " ".join(str(field.pcd_size) for field in fields)
    types = " ".join(field.pcd_type for field in fields)
    counts = " ".join(str(field.count) for field in fields)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {field_names}\n"
        f"SIZE {sizes}\n"
        f"TYPE {types}\n"
        f"COUNT {counts}\n"
        f"WIDTH {frame.width}\n"
        f"HEIGHT {frame.height}\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {frame.point_count}\n"
        "DATA binary\n"
    )
    return header.encode("ascii")


def _is_tightly_packed(
    frame: PointCloudFrame, fields: tuple[PointField, ...]
) -> bool:
    next_offset = 0
    for field in fields:
        if field.offset != next_offset:
            return False
        next_offset += field.size
    return next_offset == frame.point_step and frame.row_step == (
        frame.width * frame.point_step
    )


def _write_repacked_points(
    output: BinaryIO,
    frame: PointCloudFrame,
    fields: tuple[PointField, ...],
) -> None:
    """Remove PointCloud2 row/field padding while preserving field bytes."""

    source = memoryview(frame.data)
    for row in range(frame.height):
        row_start = row * frame.row_step
        for column in range(frame.width):
            point_start = row_start + column * frame.point_step
            for field in fields:
                start = point_start + field.offset
                output.write(source[start : start + field.size])


def write_pcd(
    output_path: str | Path,
    frame: PointCloudFrame,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one frame as an uncompressed binary PCD file.

    Field names, scalar types and counts are preserved.  Big-endian input is
    rejected because PCD binary data is conventionally little-endian and byte
    swapping arbitrary multi-count fields would require an explicit policy.
    """

    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"PCD file already exists: {destination}")
    if frame.is_bigendian:
        raise ValueError(
            f"big-endian PointCloud2 data is not supported: {frame.source_file}"
        )

    fields = _ordered_fields(frame)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            output.write(_pcd_header(frame, fields))
            if _is_tightly_packed(frame, fields):
                data_size = frame.row_step * frame.height
                output.write(frame.data[:data_size])
            else:
                _write_repacked_points(output, frame, fields)
            output.flush()
            os.fsync(output.fileno())

        os.chmod(temp_path, 0o644)
        os.replace(temp_path, destination)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return destination
