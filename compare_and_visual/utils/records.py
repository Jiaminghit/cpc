"""输入记录的通用辅助函数。

这里处理的是原始 JSONL record 的对齐、索引、时间有效性和图片选择；
具体 eval record 的构造放在 metrics/records.py。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.io import resolve_path


def aligned_key(record: dict[str, Any]) -> tuple[int, str]:
    """统一使用 pcd_timestamp + camera 作为多路记录对齐 key。"""
    return int(record["pcd_timestamp"]), str(record["camera"])


def index_records(records: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    """把 record 列表转成 key -> record 字典，便于两路 JSONL 对齐。"""
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for record in records:
        indexed[aligned_key(record)] = record
    return indexed


def is_valid_frame_record(record: dict[str, Any] | None) -> bool:
    """判断记录所属 PCD frame 是否通过整帧时间同步检查。"""
    if record is None:
        return False
    return bool(record.get("frame_valid_time_match"))


def select_comparable_keys(
    *,
    projection_by_key: dict[tuple[int, str], dict[str, Any]],
    model_by_key: dict[tuple[int, str], dict[str, Any]],
    cameras: list[str] | None,
    include_invalid: bool,
    max_records: int | None,
) -> tuple[list[tuple[int, str]], dict[str, Any]]:
    """选择正式可比较的 key，并统计被过滤掉的 key。

    默认使用 projection/model 两边 key 的交集，并过滤掉 frame invalid 的
    记录；include_invalid=True 时保留 invalid frame。
    """
    projection_keys = set(projection_by_key)
    model_keys = set(model_by_key)
    common_keys = projection_keys & model_keys
    projection_only_keys = projection_keys - model_keys
    model_only_keys = model_keys - projection_keys

    if cameras:
        camera_filter = set(cameras)
        projection_keys = {key for key in projection_keys if key[1] in camera_filter}
        model_keys = {key for key in model_keys if key[1] in camera_filter}
        common_keys = {key for key in common_keys if key[1] in camera_filter}
        projection_only_keys = {key for key in projection_only_keys if key[1] in camera_filter}
        model_only_keys = {key for key in model_only_keys if key[1] in camera_filter}

    invalid_common_keys = {
        key
        for key in common_keys
        if not (
            is_valid_frame_record(projection_by_key.get(key))
            and is_valid_frame_record(model_by_key.get(key))
        )
    }
    missing_frame_valid_keys = {
        key
        for key in common_keys
        if "frame_valid_time_match" not in projection_by_key.get(key, {})
        or "frame_valid_time_match" not in model_by_key.get(key, {})
    }
    selected_keys = common_keys if include_invalid else common_keys - invalid_common_keys
    selected_key_count_before_max_records = len(selected_keys)
    selected_keys = sorted(selected_keys)
    if max_records is not None:
        selected_keys = selected_keys[:max_records]

    warnings: list[str] = []
    if projection_only_keys:
        warnings.append(
            f"Excluded {len(projection_only_keys)} projection-only records because key selection uses intersection."
        )
    if model_only_keys:
        warnings.append(
            f"Excluded {len(model_only_keys)} model-only records because key selection uses intersection."
        )
    if invalid_common_keys and not include_invalid:
        warnings.append(
            f"Excluded {len(invalid_common_keys)} common records because frame_valid_time_match is false."
        )
    if missing_frame_valid_keys:
        warnings.append(
            f"Found {len(missing_frame_valid_keys)} common records missing frame_valid_time_match; they are treated as invalid."
        )
    if include_invalid and (projection_only_keys or model_only_keys):
        warnings.append(
            "include_invalid is enabled, but unmatched records still exist; one upstream JSONL may not include the same frames."
        )

    stats = {
        "key_selection": "intersection",
        "include_invalid": include_invalid,
        "valid_frame_only": not include_invalid,
        "projection_key_count": len(projection_keys),
        "model_key_count": len(model_keys),
        "common_key_count": len(common_keys),
        "projection_only_key_count": len(projection_only_keys),
        "model_only_key_count": len(model_only_keys),
        "invalid_common_key_count": len(invalid_common_keys),
        "missing_frame_valid_key_count": len(missing_frame_valid_keys),
        "excluded_invalid_key_count": 0 if include_invalid else len(invalid_common_keys),
        "selected_key_count_before_max_records": selected_key_count_before_max_records,
        "selected_key_count": len(selected_keys),
        "warnings": warnings,
    }
    return selected_keys, stats


def pick_rgb_image(
    dataset_root: Path,
    projection: dict[str, Any] | None,
    model: dict[str, Any] | None,
) -> Path | None:
    """从 projection/model 两侧记录里优先找到真实存在的 RGB 图片。"""
    for record in (projection, model):
        if not record:
            continue
        image = resolve_path(dataset_root, record.get("rgb_image"))
        if image and image.is_file():
            return image
        image_rel = resolve_path(dataset_root, record.get("rgb_image_rel"))
        if image_rel and image_rel.is_file():
            return image_rel
    return None
