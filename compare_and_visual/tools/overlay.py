"""单帧 overlay 绘图工具。

pipeline/projection_model_overlay.py 负责遍历和汇总；本模块只关心一条对齐记录
如何在 RGB 图上画出 reference projection 和 model detection。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from utils.classes import color_for, contrast_color
from utils.drawing import draw_corner_edges, draw_dashed_box
from utils.records import aligned_key, pick_rgb_image


def box_label(prefix: str, box: dict[str, Any]) -> str:
    """生成框标签，优先显示类别，若有 score 则附带置信度。"""
    class_name = box.get("class_name") or box.get("label_raw") or "unknown"
    score = box.get("score")
    if score is None:
        return f"{prefix} {class_name}"
    return f"{prefix} {class_name} {float(score):.2f}"


def draw_projection_boxes(image: Any, reference_boxes: list[dict[str, Any]]) -> None:
    """绘制 3D 投影参考框，使用 corners_2d 连成线框。"""
    for box in reference_boxes:
        color = color_for(box.get("class_name"))
        draw_corner_edges(
            image,
            box.get("corners_2d", []),
            box_label("REF", box),
            color,
            thickness=2,
        )


def draw_model_boxes(image: Any, model_boxes: list[dict[str, Any]], model_mask_alpha: float) -> None:
    """绘制模型检测 2D 框，使用虚线和半透明填充与参考框区分。"""
    for box in model_boxes:
        color = contrast_color(color_for(box.get("class_name")))
        draw_dashed_box(
            image,
            box.get("bbox_xyxy", [0, 0, 0, 0]),
            box_label("DINO", box),
            color,
            thickness=2,
            mask_alpha=model_mask_alpha,
        )


def overlay_record(
    *,
    dataset_root: Path,
    projection: dict[str, Any] | None,
    model: dict[str, Any] | None,
    output_path: Path,
    save_vis: bool,
    draw_projection_corners: bool,
    model_mask_alpha: float,
) -> dict[str, Any]:
    """处理单个 pcd_timestamp + camera 的 overlay 结果。"""
    base_record = projection or model
    if base_record is None:
        raise ValueError("projection and model cannot both be None")

    pcd_timestamp, camera = aligned_key(base_record)
    image_path = pick_rgb_image(dataset_root, projection, model)
    reference_boxes = projection.get("projected_boxes", []) if projection else []
    model_boxes = model.get("detections", []) if model else []

    # 即使不保存图片，也会返回计数和状态，便于做轻量 summary。
    status = "ok"
    reason = None
    if image_path is None:
        status = "skipped"
        reason = "rgb_image_missing"
    elif save_vis:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            status = "skipped"
            reason = "rgb_image_read_failed"
        else:
            # draw_projection_corners 保留给旧命令兼容；当前 reference 始终按 corners_2d 绘制。
            _ = draw_projection_corners
            draw_projection_boxes(image, reference_boxes)
            draw_model_boxes(image, model_boxes, model_mask_alpha)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(output_path), image)
            if not ok:
                status = "skipped"
                reason = "overlay_write_failed"

    return {
        "pcd_timestamp": pcd_timestamp,
        "camera": camera,
        "rgb_timestamp": base_record.get("rgb_timestamp"),
        "rgb_image": str(image_path) if image_path else None,
        "overlay_image": str(output_path) if status == "ok" and save_vis else None,
        "status": status,
        "reason": reason,
        "reference_box_count": len(reference_boxes),
        "model_box_count": len(model_boxes),
        "has_projection_record": projection is not None,
        "has_model_record": model is not None,
    }
