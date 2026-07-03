"""JSON/JSONL 与路径解析的通用工具。

这些函数不包含评估或可视化业务逻辑，只提供多个 pipeline 都会用到的基础 IO。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，并把原始行号写入 _line_no，方便排查坏记录。"""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as reader:
        for line_no, line in enumerate(reader, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_line_no"] = line_no
            records.append(record)
    return records


def write_json(path: Path, data: dict[str, Any]) -> None:
    """写缩进 JSON；自动创建父目录，方便 pipeline 直接落结果文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl_line(writer: Any, data: dict[str, Any]) -> None:
    writer.write(json.dumps(data, ensure_ascii=False) + "\n")


def resolve_path(dataset_root: Path, value: str | None) -> Path | None:
    """把记录中的图片路径解析为本地路径。

    输入可能是绝对路径、相对 dataset_root 的路径，或包含 dataset_root 名称的
    相对路径；这里统一转成可检查的 Path。
    """
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == dataset_root.name:
        return dataset_root.parent / path
    return dataset_root / path
