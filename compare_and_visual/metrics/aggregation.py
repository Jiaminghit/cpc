"""评估结果聚合工具。

record 级结果只描述一帧/一相机；本模块负责把这些结果累积成全局、按相机、
按类别的统计，并在 finalize 时计算可读指标。
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from metrics.bbox import round_float
from metrics.matching import MATCH_KEYS


def metric_ratio(numerator: int | float, denominator: int | float) -> float | None:
    """安全比例计算；分母为 0 时返回 None，避免误导性地写成 0。"""
    if denominator == 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.fmean(values)), 6)


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.median(values)), 6)


def make_empty_counts() -> dict[str, int]:
    """创建单条评估 record 使用的计数器结构。"""
    counts = {
        "projection_boxes": 0,
        "dino_boxes": 0,
        "matches": 0,
    }
    for key in MATCH_KEYS:
        counts[key] = 0
    return counts


def make_aggregate() -> dict[str, Any]:
    """创建可累积的聚合块，内部保留 IoU/距离列表直到 finalize。"""
    return {
        "records": 0,
        "records_with_projection_only": 0,
        "records_with_dino_only": 0,
        "boxes": {"projection_boxes": 0, "dino_boxes": 0},
        "matches": {key: 0 for key in MATCH_KEYS},
        "iou_values": [],
        "center_distance_px_values": [],
        "class_mismatch_pairs": defaultdict(int),
    }


def finalize_aggregate(block: dict[str, Any]) -> dict[str, Any]:
    """把聚合块转成最终 JSON 结构，并计算 pseudo precision/recall/F1。

    这里把 tp + class_mismatch 视为几何上找到了对应目标，因此用于 pseudo
    precision/recall 的分子；类别错误会在 matches 和 class_mismatch_pairs 中保留。
    """
    matches = dict(block["matches"])
    geometry_tp = matches["tp"] + matches["class_mismatch"]
    precision = metric_ratio(geometry_tp, geometry_tp + matches["unmatched_projection"])
    recall = metric_ratio(geometry_tp, geometry_tp + matches["unmatched_dino"])
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = round(2.0 * precision * recall / (precision + recall), 6)
    return {
        "records": block["records"],
        "records_with_projection_only": block["records_with_projection_only"],
        "records_with_dino_only": block["records_with_dino_only"],
        "boxes": dict(block["boxes"]),
        "matches": matches,
        "metrics": {
            "pseudo_precision": precision,
            "pseudo_recall": recall,
            "pseudo_f1": f1,
            "mean_iou_matched": mean_or_none(block["iou_values"]),
            "median_iou_matched": median_or_none(block["iou_values"]),
            "mean_center_distance_px": mean_or_none(block["center_distance_px_values"]),
        },
        "class_mismatch_pairs": dict(sorted(block["class_mismatch_pairs"].items())),
    }


def update_aggregate_from_record(block: dict[str, Any], record: dict[str, Any]) -> None:
    """把一条 record 的 counts/matches 累积进全局或相机级聚合块。"""
    counts = record["counts"]
    block["records"] += 1
    if counts["projection_boxes"] > 0 and counts["dino_boxes"] == 0:
        block["records_with_projection_only"] += 1
    if counts["dino_boxes"] > 0 and counts["projection_boxes"] == 0:
        block["records_with_dino_only"] += 1
    block["boxes"]["projection_boxes"] += counts["projection_boxes"]
    block["boxes"]["dino_boxes"] += counts["dino_boxes"]
    for key in MATCH_KEYS:
        block["matches"][key] += counts[key]
    for match in record["matches"]:
        block["iou_values"].append(float(match["iou"]))
        block["center_distance_px_values"].append(float(match["center_distance_px"]))
        if match["match_status"] == "class_mismatch":
            projection_class = match["projection"].get("class_name") or "unknown"
            dino_class = match["dino"].get("class_name") or "unknown"
            block["class_mismatch_pairs"][f"{projection_class}->{dino_class}"] += 1


def update_class_aggregate(
    blocks: dict[str, dict[str, Any]],
    record: dict[str, Any],
) -> None:
    """按类别累积指标。

    匹配对以 projection 类别归属；unmatched projection/DINO 分别按自己携带的
    class_name 归属。touched_classes 用于统计“该类别出现过的记录数”。
    """
    touched_classes: set[str] = set()
    for match in record["matches"]:
        projection_class = match["projection"].get("class_name") or "unknown"
        touched_classes.add(projection_class)
        block = blocks[projection_class]
        block["boxes"]["projection_boxes"] += 1
        block["boxes"]["dino_boxes"] += 1
        block["matches"][match["match_status"]] += 1
        block["iou_values"].append(float(match["iou"]))
        block["center_distance_px_values"].append(float(match["center_distance_px"]))
        if match["match_status"] == "class_mismatch":
            dino_class = match["dino"].get("class_name") or "unknown"
            block["class_mismatch_pairs"][f"{projection_class}->{dino_class}"] += 1

    for item in record["unmatched_projection_boxes"]:
        class_name = item["projection"].get("class_name") or "unknown"
        touched_classes.add(class_name)
        block = blocks[class_name]
        block["boxes"]["projection_boxes"] += 1
        block["matches"]["unmatched_projection"] += 1

    for item in record["unmatched_dino_boxes"]:
        class_name = item["dino"].get("class_name") or "unknown"
        touched_classes.add(class_name)
        block = blocks[class_name]
        block["boxes"]["dino_boxes"] += 1
        block["matches"]["unmatched_dino"] += 1

    for class_name in touched_classes:
        blocks[class_name]["records"] += 1
