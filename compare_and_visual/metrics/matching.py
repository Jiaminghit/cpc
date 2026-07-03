"""projection bbox 与 DINO bbox 的匹配逻辑。

当前使用 greedy matching：先枚举有重叠的候选对，按 IoU 从高到低排序，
再保证每个 projection/DINO box 最多只被匹配一次。
"""

from __future__ import annotations

from typing import Any

from metrics.bbox import (
    area_ratio_projection_over_dino,
    bbox_iou,
    center_distance_px,
    round_float,
)


MATCH_KEYS = [
    "tp",
    "class_mismatch",
    "low_iou_matched",
    "weak_overlap_matched",
    "unmatched_projection",
    "unmatched_dino",
]


def match_status_for(iou: float, class_equal: bool, iou_threshold: float, low_iou_threshold: float) -> str:
    """根据 IoU 和类别是否一致，把已匹配候选对划分为 TP/类别错/低 IoU/弱重叠。"""
    if iou >= iou_threshold and class_equal:
        return "tp"
    if iou >= iou_threshold and not class_equal:
        return "class_mismatch"
    if iou >= low_iou_threshold:
        return "low_iou_matched"
    return "weak_overlap_matched"


def best_iou_for_unmatched(
    source_bbox: list[float],
    other_boxes: list[dict[str, Any]],
    other_id_key: str,
) -> tuple[float | None, str | None]:
    """给未匹配框补充它与另一侧所有框的最大 IoU，便于后续排查边界案例。"""
    best_iou = 0.0
    best_id = None
    for other in other_boxes:
        iou = bbox_iou(source_bbox, other["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_id = other["raw"].get(other_id_key)
    return (round(best_iou, 6), best_id) if best_id is not None else (None, None)


def greedy_match(
    projection_boxes: list[dict[str, Any]],
    dino_boxes: list[dict[str, Any]],
    *,
    iou_threshold: float,
    low_iou_threshold: float,
    class_aware_matching: bool,
    pcd_timestamp: int,
    camera: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """执行一帧内的 greedy bbox 匹配，并返回 matches 与两侧 unmatched 明细。"""
    candidates = []
    for projection in projection_boxes:
        for dino in dino_boxes:
            class_equal = projection["raw"].get("class_name") == dino["raw"].get("class_name")
            if class_aware_matching and not class_equal:
                continue
            iou = bbox_iou(projection["bbox"], dino["bbox"])
            if iou <= 0.0:
                continue
            distance = center_distance_px(projection["bbox"], dino["bbox"])
            # 候选对保留原始对象和几何指标，后续排序后再决定一对一匹配关系。
            candidates.append(
                {
                    "projection_index": projection["index"],
                    "dino_index": dino["index"],
                    "projection": projection,
                    "dino": dino,
                    "iou": iou,
                    "center_distance_px": distance,
                    "area_ratio_projection_over_dino": area_ratio_projection_over_dino(
                        projection["bbox"], dino["bbox"]
                    ),
                    "class_equal": class_equal,
                }
            )
    # 优先选择 IoU 最大的候选；IoU 相同时，中心距离更近的候选排在前面。
    candidates.sort(key=lambda item: (-item["iou"], item["center_distance_px"]))

    used_projection: set[int] = set()
    used_dino: set[int] = set()
    matches = []
    for candidate in candidates:
        projection_index = candidate["projection_index"]
        dino_index = candidate["dino_index"]
        if projection_index in used_projection or dino_index in used_dino:
            continue
        # 一旦某个框被使用，就不再参与后续候选，保证一对一匹配。
        used_projection.add(projection_index)
        used_dino.add(dino_index)
        status = match_status_for(
            candidate["iou"],
            candidate["class_equal"],
            iou_threshold,
            low_iou_threshold,
        )
        projection = candidate["projection"]
        dino = candidate["dino"]
        match_id = f"match:{pcd_timestamp}:{camera}:{len(matches)}"
        matches.append(
            {
                "match_id": match_id,
                "match_status": status,
                "iou": round(candidate["iou"], 6),
                "center_distance_px": round(candidate["center_distance_px"], 6),
                "area_ratio_projection_over_dino": round_float(
                    candidate["area_ratio_projection_over_dino"]
                ),
                "class_equal": candidate["class_equal"],
                "projection": projection["compact"],
                "dino": dino["compact"],
            }
        )

    # 没进入 used_* 的框会单独输出，附带 best_iou 辅助人工分析。
    unmatched_projection = []
    for projection in projection_boxes:
        if projection["index"] in used_projection:
            continue
        best_iou, best_det_id = best_iou_for_unmatched(projection["bbox"], dino_boxes, "det_id")
        unmatched_projection.append(
            {
                "match_status": "unmatched_projection",
                "best_iou": best_iou,
                "best_dino_det_id": best_det_id,
                "projection": projection["compact"],
            }
        )

    unmatched_dino = []
    for dino in dino_boxes:
        if dino["index"] in used_dino:
            continue
        best_iou, best_projection_id = best_iou_for_unmatched(dino["bbox"], projection_boxes, "projection_id")
        unmatched_dino.append(
            {
                "match_status": "unmatched_dino",
                "best_iou": best_iou,
                "best_projection_id": best_projection_id,
                "dino": dino["compact"],
            }
        )

    return matches, unmatched_projection, unmatched_dino
