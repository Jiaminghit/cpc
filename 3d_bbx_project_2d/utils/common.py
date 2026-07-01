"""通用 IO 与数值格式化工具。

任务角色：
    为各类投影脚本提供最基础的 JSON 读取、数据集路径解析和
    浮点数输出规范化能力，避免不同脚本各自实现导致输出格式漂移。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# 工作角色：读取 JSON 文件。
# 输入输出：输入 JSON 文件路径，输出解析后的 dict。
# 实现思路：统一使用 UTF-8 打开文件，并通过 json.load 转成 Python 对象。
def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


# 工作角色：把 index/meta 中记录的路径解析成真实文件路径。
# 输入输出：输入数据集根目录和可能为相对/绝对路径的字符串，输出 Path 或 None。
# 实现思路：绝对路径直接返回；带数据集目录名前缀的相对路径从父目录解析；其余路径拼到 dataset_root 下。
def resolve_path(dataset_root: Path, rel_path: str | None) -> Path | None:
    if not rel_path:
        return None
    path = Path(rel_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == dataset_root.name:
        return dataset_root.parent / path
    return dataset_root / path


# 工作角色：统一结构化输出中的浮点数精度。
# 输入输出：输入数值，输出保留 6 位小数的 float。
# 实现思路：先转成 Python float，再调用 round，避免 numpy 标量写 JSON 时出现类型问题。
def round_float(value: float) -> float:
    return round(float(value), 6)


# 工作角色：将 numpy 数组转换为适合写入 JSON 的列表。
# 输入输出：输入 numpy.ndarray，输出四舍五入后的嵌套 list。
# 实现思路：转 float64 后统一保留 6 位小数，再调用 tolist 去掉 numpy 类型。
def array_to_rounded_list(values: np.ndarray) -> list[Any]:
    return np.round(values.astype(np.float64), 6).tolist()
