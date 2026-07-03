from __future__ import annotations

"""PCD 点云读取工具。

当前 ROI 过滤只需要 x/y/z 三列，因此这里会解析 PCD header，
再从 ascii 或 binary 数据中提取有限的 xyz 点。
"""

from pathlib import Path
from typing import Any

import numpy as np


def pcd_numpy_type(field_type: str, size: int) -> str:
    """把 PCD 字段类型/字节数转换为 numpy dtype 字符串。"""
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
    """读取 PCD header，并返回二进制/文本数据区起始偏移。"""
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
    """读取 PCD 文件中的 x/y/z 点，输出 N x 3 float64 数组。"""
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
        # ascii PCD 可直接用 loadtxt 读取；只取 x/y/z 三列。
        data = np.loadtxt(path, comments="#", skiprows=len(header) + 1)
        field_index = {name: idx for idx, name in enumerate(fields)}
        return np.asarray(
            data[:, [field_index["x"], field_index["y"], field_index["z"]]],
            dtype=np.float64,
        )

    if data_kind != "binary":
        raise ValueError(f"unsupported PCD DATA format: {data_kind}")

    dtype_fields: list[tuple[str, Any]] = []
    # binary PCD 需要根据 header 中每个字段的 type/size/count 构造结构化 dtype。
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
    # 去掉 NaN/Inf 点，避免后续投影和 median 计算污染结果。
    finite_mask = np.isfinite(xyz).all(axis=1)
    return xyz[finite_mask]
