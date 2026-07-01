"""PCD file reading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def pcd_type_to_dtype(type_code: str, size: int) -> str:
    if type_code == "F" and size == 4:
        return "<f4"
    if type_code == "F" and size == 8:
        return "<f8"
    if type_code == "U" and size == 1:
        return "<u1"
    if type_code == "U" and size == 2:
        return "<u2"
    if type_code == "U" and size == 4:
        return "<u4"
    if type_code == "U" and size == 8:
        return "<u8"
    if type_code == "I" and size == 1:
        return "<i1"
    if type_code == "I" and size == 2:
        return "<i2"
    if type_code == "I" and size == 4:
        return "<i4"
    if type_code == "I" and size == 8:
        return "<i8"
    raise ValueError(f"unsupported PCD field type/size: {type_code}{size}")


def read_binary_pcd_xyzi(pcd_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    header: dict[str, list[str]] = {}
    with pcd_path.open("rb") as reader:
        while True:
            line = reader.readline()
            if not line:
                raise ValueError(f"PCD header has no DATA line: {pcd_path}")
            decoded = line.decode("ascii", errors="strict").strip()
            if not decoded:
                continue
            parts = decoded.split()
            key = parts[0].upper()
            header[key] = parts[1:]
            if key == "DATA":
                if len(parts) < 2 or parts[1].lower() != "binary":
                    raise ValueError(f"only binary PCD is supported: {pcd_path}")
                break

        fields = header["FIELDS"]
        sizes = [int(item) for item in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(item) for item in header.get("COUNT", ["1"] * len(fields))]
        points = int(header.get("POINTS", header.get("WIDTH", ["0"]))[0])

        dtype_fields: list[tuple[str, Any]] = []
        for field, size, type_code, count in zip(fields, sizes, types, counts):
            dtype = np.dtype(pcd_type_to_dtype(type_code, size))
            if count == 1:
                dtype_fields.append((field, dtype))
            else:
                dtype_fields.append((field, dtype, (count,)))
        point_dtype = np.dtype(dtype_fields)
        cloud = np.fromfile(reader, dtype=point_dtype, count=points)

    missing = {"x", "y", "z"} - set(cloud.dtype.names or [])
    if missing:
        raise ValueError(f"PCD missing required fields {sorted(missing)}: {pcd_path}")

    xyz = np.stack(
        [
            cloud["x"].astype(np.float64),
            cloud["y"].astype(np.float64),
            cloud["z"].astype(np.float64),
        ],
        axis=1,
    )
    intensity = (
        cloud["intensity"].astype(np.float64)
        if cloud.dtype.names is not None and "intensity" in cloud.dtype.names
        else None
    )
    return xyz, intensity
