from __future__ import annotations

"""通用 IO、路径解析与数值格式化工具。"""

import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON 文件并返回 dict。"""
    return json.loads(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    """`read_json` 的语义别名，兼容不同脚本中的命名习惯。"""
    return read_json(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """以统一缩进和 UTF-8 编码写出 JSON 文件。"""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件，忽略空行。"""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_dataset_path(dataset_root: Path, value: str | None) -> Path | None:
    """把数据集中记录的相对/绝对路径解析为本机真实路径。"""
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == dataset_root.name:
        return dataset_root.parent / path
    return dataset_root / path


def resolve_path(dataset_root: Path, value: str | None) -> Path | None:
    """`resolve_dataset_path` 的短别名，供 ROI pipeline 读取 pcd/calib/rgb 路径。"""
    return resolve_dataset_path(dataset_root, value)


def resolve_project_path(value: str) -> Path:
    """解析项目内资源路径，例如 GroundingDINO config 和 checkpoint。"""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def round_float(value: float) -> float:
    """统一 JSON 输出中的浮点数精度。"""
    return round(float(value), 6)


def array_to_rounded_list(values: np.ndarray | None) -> list[Any] | None:
    """把 numpy 数组转换成可 JSON 序列化的列表。"""
    if values is None:
        return None
    return np.round(values.astype(np.float64), 6).tolist()
