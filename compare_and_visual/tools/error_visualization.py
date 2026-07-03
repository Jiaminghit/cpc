"""评估错误样例可视化工具。

与普通 overlay 不同，这里只在记录存在非 TP 状态时输出图片，并按错误类型
拆目录保存，方便集中查看 class_mismatch、low_iou、unmatched 等问题。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from metrics.bbox import normalize_bbox_xyxy
from utils.records import pick_rgb_image


VIS_COLORS = {
    "tp": (60, 220, 60),
    "class_mismatch": (255, 80, 255),
    "low_iou_matched": (40, 220, 255),
    "weak_overlap_matched": (40, 180, 255),
    "unmatched_projection": (40, 40, 255),
    "unmatched_dino": (255, 120, 40),
}


def import_cv2() -> Any:
    """延迟导入 cv2，让不保存 error vis 的评估流程无需依赖 OpenCV。"""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("--save-error-vis requires cv2/opencv-python in the active environment") from exc
    return cv2


def draw_label(cv2: Any, image: Any, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    """在错误图上绘制标签，黑底保证不同图像背景下可读。"""
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = int(min(max(origin[0], 0), max(0, width - text_w - 2)))
    y = int(min(max(origin[1], text_h + baseline + 2), height - 2))
    cv2.rectangle(
        image,
        (x, y - text_h - baseline - 3),
        (x + text_w + 4, y + baseline + 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(image, text, (x + 2, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_box(
    cv2: Any,
    image: Any,
    bbox: list[Any],
    label: str,
    color: tuple[int, int, int],
    *,
    dashed: bool,
) -> None:
    """绘制错误图中的 bbox；projection 用实线，DINO 用虚线。"""
    height, width = image.shape[:2]
    normalized = normalize_bbox_xyxy(bbox, width, height)
    if normalized is None:
        return
    x1, y1, x2, y2 = [int(round(value)) for value in normalized]
    if dashed:
        dash = 14
        gap = 8
        for start, end in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
            sx, sy = start
            ex, ey = end
            length = math.hypot(ex - sx, ey - sy)
            if length <= 0:
                continue
            vx = (ex - sx) / length
            vy = (ey - sy) / length
            pos = 0.0
            while pos < length:
                seg_end = min(pos + dash, length)
                p1 = (int(round(sx + vx * pos)), int(round(sy + vy * pos)))
                p2 = (int(round(sx + vx * seg_end)), int(round(sy + vy * seg_end)))
                cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
                pos += dash + gap
        draw_label(cv2, image, label, (x1, min(height - 2, y2 + 18)), color)
    else:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        draw_label(cv2, image, label, (x1, max(0, y1 - 6)), color)


def save_error_visualization(
    *,
    cv2: Any,
    dataset_root: Path,
    output_dir: Path,
    record: dict[str, Any],
    projection_record: dict[str, Any] | None,
    dino_record: dict[str, Any] | None,
) -> list[str]:
    """为一条评估 record 保存错误可视化图片。

    返回实际写出的路径列表；如果该记录只有 TP，或者图片不存在/读取失败，
    则返回空列表。
    """
    statuses = {
        match["match_status"]
        for match in record["matches"]
        if match["match_status"] != "tp"
    }
    if record["unmatched_projection_boxes"]:
        statuses.add("unmatched_projection")
    if record["unmatched_dino_boxes"]:
        statuses.add("unmatched_dino")
    if not statuses:
        return []

    image_path = pick_rgb_image(dataset_root, projection_record, dino_record)
    if image_path is None:
        return []
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return []

    # 匹配对按状态着色；非 TP 状态会被写入对应 error_vis/<status>/ 目录。
    for match in record["matches"]:
        status = match["match_status"]
        color = VIS_COLORS.get(status, (220, 220, 220))
        draw_box(
            cv2,
            image,
            match["projection"].get("bbox_xyxy") or [0, 0, 0, 0],
            f"REF {status} {match['iou']:.2f}",
            color,
            dashed=False,
        )
        draw_box(
            cv2,
            image,
            match["dino"].get("bbox_xyxy") or [0, 0, 0, 0],
            f"DINO {status}",
            color,
            dashed=True,
        )

    # unmatched 两侧单独绘制，并把 best_iou 写进标签辅助排查。
    for item in record["unmatched_projection_boxes"]:
        color = VIS_COLORS["unmatched_projection"]
        draw_box(
            cv2,
            image,
            item["projection"].get("bbox_xyxy") or [0, 0, 0, 0],
            f"REF unmatched best={item.get('best_iou')}",
            color,
            dashed=False,
        )

    for item in record["unmatched_dino_boxes"]:
        color = VIS_COLORS["unmatched_dino"]
        draw_box(
            cv2,
            image,
            item["dino"].get("bbox_xyxy") or [0, 0, 0, 0],
            f"DINO unmatched best={item.get('best_iou')}",
            color,
            dashed=True,
        )

    written = []
    pcd_timestamp = record["pcd_timestamp"]
    camera = record["camera"]
    for status in sorted(statuses):
        path = output_dir / "error_vis" / status / camera / f"{pcd_timestamp}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(path), image):
            written.append(str(path))
    return written
