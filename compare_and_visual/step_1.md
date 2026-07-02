# Step 1: ROI 内 2D 一致性评价实施步骤

本文档用于指导后续实现 `compare_and_visual/evaluate_roi_projection_vs_dino.py`。当前阶段只完成 `week_2_plan.md` 中的第一层目标：在 ROI 内比较 3D 检测框投影结果与 GroundingDINO ROI baseline 的 2D bbox 一致性，并输出可量化的匹配结果和统计指标。

## 1. 目标边界

核心定义：

```text
3D 投影框 = 被评价对象
DINO ROI 检测框 = 2D baseline / pseudo reference
评价结论 = 2D 一致性结果，不等价于人工真值
```

第一层只回答这些问题：

```text
1. 每个 ROI 内 3D 投影框是否能在同图 DINO ROI baseline 中找到几何匹配。
2. 每个 ROI 内 DINO baseline 框是否被 3D 投影框覆盖到。
3. 匹配上的框 IoU、中心偏移、面积比例是否合理。
4. 按相机、类别、全局输出 pseudo precision / recall / F1。
```

第一层暂不处理：

```text
1. 跨帧 Track_id 错误。
2. 尺寸跳变、方向跳变等时序错误。
3. BEV 空间重叠错误。
4. 人工真值判定。
```

## 2. 当前已打通输入

数据集：

```text
/home/c64508/桌面/dataset/2069758074335653889
```

3D ROI 投影结果：

```text
/home/c64508/桌面/dataset/2069758074335653889/vis_projection_newstruct_final_roi/projections_aligned.jsonl
```

来源脚本：

```text
3d_bbx_project_2d/project_boxes_to_images_withROI.py
```

DINO ROI baseline 结果：

```text
/home/c64508/桌面/dataset/2069758074335653889/aligned_grounding_dino_b_roi/detections_aligned_roi_only.jsonl
```

来源脚本：

```text
2d_detect/run_aligned_model_detection.py
2d_detect/filter_dino_detections_by_lidar_roi.py
```

已有叠加可视化结果：

```text
/home/c64508/桌面/dataset/2069758074335653889/compare_grounding_dino_b_roi_projection_roi
```

来源脚本：

```text
compare_and_visual/visualize_projection_vs_model.py
```

## 3. 新增脚本与默认输出

建议新增脚本：

```text
compare_and_visual/evaluate_roi_projection_vs_dino.py
```

默认输出目录：

```text
/home/c64508/桌面/dataset/2069758074335653889/compare_grounding_dino_b_roi_projection_qc
```

输出文件：

```text
compare_grounding_dino_b_roi_projection_qc/
  manifest.json
  matches_aligned.jsonl
  metrics_summary.json
  metrics_by_camera.json
  metrics_by_class.json
  unmatched_projection_boxes.jsonl
  unmatched_dino_boxes.jsonl
```

第一层可以先不输出 `qc_error_records.jsonl`。该文件属于第二层“映射到质检错误类型”，但 `matches_aligned.jsonl` 中应保留足够字段，便于后续直接生成 QC 疑似问题清单。

## 4. CLI 参数设计

第一版建议参数：

```bash
python3 compare_and_visual/evaluate_roi_projection_vs_dino.py \
  --dataset-root /home/c64508/桌面/dataset/2069758074335653889 \
  --projection-jsonl /home/c64508/桌面/dataset/2069758074335653889/vis_projection_newstruct_final_roi/projections_aligned.jsonl \
  --dino-jsonl /home/c64508/桌面/dataset/2069758074335653889/aligned_grounding_dino_b_roi/detections_aligned_roi_only.jsonl \
  --output-dir /home/c64508/桌面/dataset/2069758074335653889/compare_grounding_dino_b_roi_projection_qc \
  --iou-threshold 0.5 \
  --low-iou-threshold 0.3
```

可选参数：

```text
--max-records N
--cameras camera_front_wide camera_front_left ...
--class-aware-matching
--allow-class-mismatch
--valid-time-only
--save-error-vis
```

建议默认：

```text
iou_threshold = 0.5
low_iou_threshold = 0.3
class_aware_matching = false
allow_class_mismatch = true
valid_time_only = false
```

原因：第一层重点是几何一致性。类别错误应作为匹配后的属性差异记录，而不是一开始就阻断几何匹配。

## 5. 读取与对齐

两个 JSONL 都按以下主键对齐：

```text
(pcd_timestamp, camera)
```

投影记录读取字段：

```text
record.pcd_timestamp
record.camera
record.rgb_timestamp
record.rgb_image
record.rgb_image_rel
record.image_width
record.image_height
record.frame_valid_time_match
record.valid_time_match
record.projected_boxes[]
```

投影框读取字段：

```text
projected_boxes[].projection_id
projected_boxes[].object_id
projected_boxes[].class_name
projected_boxes[].class_id
projected_boxes[].score
projected_boxes[].bbox_xyxy
projected_boxes[].bbox_3d.center_ego
projected_boxes[].bbox_3d.size_lwh
projected_boxes[].bbox_3d.heading
projected_boxes[].roi.in_roi
projected_boxes[].roi.longitudinal_m
projected_boxes[].roi.lateral_m
```

DINO 记录读取字段：

```text
record.pcd_timestamp
record.camera
record.rgb_timestamp
record.rgb_image
record.rgb_image_rel
record.image_width
record.image_height
record.frame_valid_time_match
record.valid_time_match
record.detections[]
```

DINO 框读取字段：

```text
detections[].det_id
detections[].class_name
detections[].class_id
detections[].score
detections[].bbox_xyxy
detections[].lidar_roi.in_roi
detections[].lidar_roi.longitudinal_m
detections[].lidar_roi.lateral_m
```

注意：

```text
1. 当前 projection_jsonl 已使用 --roi-filter，projected_boxes 理论上均为 ROI 内。
2. 当前 dino_jsonl 使用 detections_aligned_roi_only.jsonl，detections 理论上均为 ROI 内。
3. 脚本仍应检查 roi.in_roi / lidar_roi.in_roi，发现 false 或 null 时写入 manifest/summary 的 warning。
4. valid_time_only 默认不启用，因为当前已有样例存在 valid_time_match=false；若启用，则只统计 valid_time_match=true 的记录。
```

## 6. Bbox 预处理

统一使用裁剪后的 2D bbox：

```text
projection: projected_boxes[].bbox_xyxy
DINO: detections[].bbox_xyxy
```

每个 bbox 需要校验：

```text
1. 长度为 4。
2. x2 > x1 且 y2 > y1。
3. 坐标为有限数值。
4. 坐标可按 image_width/image_height 做轻量 clip，但不要改变原始字段；输出中同时记录原始 bbox 和用于计算的 bbox。
```

建议实现工具函数：

```text
normalize_bbox_xyxy(bbox, image_width, image_height)
bbox_area(bbox)
bbox_iou(box_a, box_b)
bbox_center(box)
center_distance_px(box_a, box_b)
area_ratio(box_a, box_b)
```

`area_ratio` 建议定义为：

```text
projection_area / dino_area
```

若 dino_area 为 0，则记为 null。

## 7. IoU 匹配策略

每张图独立匹配：

```text
projection_boxes = ROI 内 3D 投影框
dino_boxes = ROI 内 DINO baseline 框
```

生成所有候选对：

```text
for each projection_box:
  for each dino_box:
    compute iou
    compute center_distance_px
    compute area_ratio
    compute class_equal
```

候选过滤：

```text
第一版建议保留 iou > 0 的所有候选。
若启用 --class-aware-matching，则只保留 class_name 相同的候选。
默认不启用 class-aware，以便发现 class_mismatch。
```

一对一匹配：

```text
1. 按 IoU 从高到低排序候选。
2. IoU 相同或非常接近时，中心距离更小者优先。
3. 贪心选择未被占用的 projection_box 和 dino_box。
4. 每个 projection_box 最多匹配一个 dino_box。
5. 每个 dino_box 最多匹配一个 projection_box。
```

选择贪心匹配的原因：

```text
1. 第一版易实现、易解释。
2. 同一图内目标数量通常不大。
3. 输出候选 IoU 矩阵后，后续可替换为 Hungarian matching。
```

可选升级：

```text
若后续发现密集目标场景中贪心匹配不稳定，再增加 --matching-method greedy/hungarian。
```

## 8. 匹配状态定义

对每个成功建立一对一关系的匹配，按 IoU 和类别给状态：

```text
iou >= iou_threshold 且 class_equal:
  match_status = "tp"

iou >= iou_threshold 且 not class_equal:
  match_status = "class_mismatch"

low_iou_threshold <= iou < iou_threshold:
  match_status = "low_iou_matched"

0 < iou < low_iou_threshold:
  match_status = "weak_overlap_matched"
```

未匹配 3D 投影框：

```text
match_status = "unmatched_projection"
含义：ROI 内 3D 投影框没有匹配到 DINO baseline，后续可映射为多标疑似或 baseline 漏检。
```

未匹配 DINO 框：

```text
match_status = "unmatched_dino"
含义：ROI 内 DINO baseline 没有匹配到 3D 投影框，后续可映射为漏标疑似或 baseline 误检。
```

第一层指标口径建议：

```text
TP = match_status in ["tp", "class_mismatch"] 且 iou >= iou_threshold
FP_projection = unmatched_projection
FN_projection = unmatched_dino
low_iou = low_iou_matched + weak_overlap_matched
class_mismatch = class_mismatch
```

说明：`class_mismatch` 几何上匹配成功，所以计入几何 TP；类别错误单独统计。

## 9. matches_aligned.jsonl 结构

每行对应一个 `(pcd_timestamp, camera)` 记录，建议结构：

```json
{
  "schema_version": "roi_projection_dino_matches.v1",
  "dataset_id": "2069758074335653889",
  "pcd_timestamp": 1782286937435038208,
  "camera": "camera_front_wide",
  "rgb_timestamp": 1782286937627422849,
  "rgb_image": "/abs/path/to/image.jpg",
  "valid_time_match": false,
  "frame_valid_time_match": false,
  "image_width": 3840,
  "image_height": 2160,
  "counts": {
    "projection_boxes": 1,
    "dino_boxes": 1,
    "matches": 1,
    "tp": 1,
    "class_mismatch": 0,
    "low_iou_matched": 0,
    "weak_overlap_matched": 0,
    "unmatched_projection": 0,
    "unmatched_dino": 0
  },
  "matches": [
    {
      "match_id": "match:1782286937435038208:camera_front_wide:0",
      "match_status": "tp",
      "iou": 0.72,
      "center_distance_px": 8.3,
      "area_ratio_projection_over_dino": 0.88,
      "class_equal": true,
      "projection": {
        "projection_id": "projection:1782286937435038208:camera_front_wide:0",
        "object_id": "1",
        "class_name": "Car",
        "class_id": 0,
        "score": 0.66748,
        "bbox_xyxy": [1917.70, 995.78, 2000.38, 1065.15],
        "roi": {
          "in_roi": true,
          "longitudinal_m": 52.85831,
          "lateral_m": -1.865252
        }
      },
      "dino": {
        "det_id": "grounding_dino_b:1782286937435038208:camera_front_wide:0",
        "class_name": "Car",
        "class_id": 0,
        "score": 0.694069,
        "bbox_xyxy": [1911.01, 997.64, 1994.37, 1075.76],
        "lidar_roi": {
          "in_roi": true,
          "longitudinal_m": 50.651789,
          "lateral_m": -1.826913
        }
      }
    }
  ],
  "unmatched_projection_boxes": [],
  "unmatched_dino_boxes": []
}
```

字段保留原则：

```text
1. bbox 使用原始输入 bbox_xyxy。
2. iou 使用 normalize 后 bbox 计算。
3. projection 与 dino 子对象保留后续 QC 可追踪所需字段。
4. 不在第一层写死“多标/漏标”等人工质检结论，只写 match_status。
```

## 10. 汇总指标

`metrics_summary.json` 建议包含：

```json
{
  "schema_version": "roi_projection_dino_metrics_summary.v1",
  "dataset_root": "/home/c64508/桌面/dataset/2069758074335653889",
  "projection_jsonl": ".../projections_aligned.jsonl",
  "dino_jsonl": ".../detections_aligned_roi_only.jsonl",
  "output_dir": ".../compare_grounding_dino_b_roi_projection_qc",
  "params": {
    "iou_threshold": 0.5,
    "low_iou_threshold": 0.3,
    "matching_method": "greedy",
    "class_aware_matching": false,
    "valid_time_only": false
  },
  "records": {
    "projection_records": 0,
    "dino_records": 0,
    "aligned_records": 0,
    "records_with_projection_only": 0,
    "records_with_dino_only": 0
  },
  "boxes": {
    "projection_boxes": 0,
    "dino_boxes": 0
  },
  "matches": {
    "tp": 0,
    "class_mismatch": 0,
    "low_iou_matched": 0,
    "weak_overlap_matched": 0,
    "unmatched_projection": 0,
    "unmatched_dino": 0
  },
  "metrics": {
    "pseudo_precision": 0.0,
    "pseudo_recall": 0.0,
    "pseudo_f1": 0.0,
    "mean_iou_matched": 0.0,
    "median_iou_matched": 0.0,
    "mean_center_distance_px": 0.0
  },
  "warnings": []
}
```

指标公式：

```text
geometry_tp = tp + class_mismatch
pseudo_precision = geometry_tp / (geometry_tp + unmatched_projection)
pseudo_recall = geometry_tp / (geometry_tp + unmatched_dino)
pseudo_f1 = 2 * precision * recall / (precision + recall)
```

边界处理：

```text
分母为 0 时指标记为 null，不要记为 0。
mean/median 没有样本时记为 null。
```

`metrics_by_camera.json`：

```text
按 camera 聚合 summary 中 boxes、matches、metrics。
```

`metrics_by_class.json`：

```text
按 projection.class_name 聚合 projection 侧指标。
未匹配 DINO 框按 dino.class_name 计入 FN 侧指标。
class_mismatch 同时记录 projection_class 与 dino_class pair。
```

## 11. unmatched 明细

`unmatched_projection_boxes.jsonl` 每行一个未匹配投影框：

```json
{
  "pcd_timestamp": 1782286937435038208,
  "camera": "camera_front_wide",
  "rgb_image": "/abs/path/to/image.jpg",
  "match_status": "unmatched_projection",
  "best_iou": 0.12,
  "best_dino_det_id": "grounding_dino_b:...",
  "projection": {}
}
```

`unmatched_dino_boxes.jsonl` 每行一个未匹配 DINO 框：

```json
{
  "pcd_timestamp": 1782286937435038208,
  "camera": "camera_front_wide",
  "rgb_image": "/abs/path/to/image.jpg",
  "match_status": "unmatched_dino",
  "best_iou": 0.08,
  "best_projection_id": "projection:...",
  "dino": {}
}
```

`best_iou` 用于区分完全无重叠和有一点重叠但被其他框占用或低于候选策略的情况。

## 12. 可视化可作为第二步增强

第一版评价脚本可以先只输出 JSON/JSONL。确认指标正确后，再增加：

```text
--save-error-vis
```

建议输出：

```text
error_vis/
  unmatched_projection/<camera>/<pcd_timestamp>.jpg
  unmatched_dino/<camera>/<pcd_timestamp>.jpg
  low_iou_matched/<camera>/<pcd_timestamp>.jpg
  class_mismatch/<camera>/<pcd_timestamp>.jpg
```

画图约定：

```text
3D 投影框: 实线
DINO baseline: 虚线或半透明填充
TP: 绿色
low_iou / weak_overlap: 黄色
unmatched_projection: 红色
unmatched_dino: 蓝色
class_mismatch: 紫色或品红
```

可复用 `compare_and_visual/visualize_projection_vs_model.py` 的：

```text
read_jsonl
write_json
aligned_key
resolve_path
pick_rgb_image
draw_label
draw_projected_box
draw_model_box
```

## 13. 实现顺序

建议按以下顺序写代码：

```text
1. 新建 evaluate_roi_projection_vs_dino.py，先完成 CLI、read_jsonl、index_records。
2. 读取两个输入文件，按 (pcd_timestamp, camera) 取并集 key。
3. 实现 bbox 工具函数和 IoU 单元级自测样例。
4. 对单条 record 生成 projection_boxes / dino_boxes 的标准化列表。
5. 实现候选 pair 生成与 greedy 一对一匹配。
6. 生成每条 record 的 matches / unmatched_projection / unmatched_dino。
7. 写 matches_aligned.jsonl。
8. 聚合全局、camera、class 指标。
9. 写 metrics_summary.json、metrics_by_camera.json、metrics_by_class.json。
10. 抽样核对已有 overlay 可视化，确认低 IoU / unmatched 是否符合直觉。
```

## 14. 验证命令

小样本试跑：

```bash
python3 compare_and_visual/evaluate_roi_projection_vs_dino.py \
  --dataset-root /home/c64508/桌面/dataset/2069758074335653889 \
  --projection-jsonl /home/c64508/桌面/dataset/2069758074335653889/vis_projection_newstruct_final_roi/projections_aligned.jsonl \
  --dino-jsonl /home/c64508/桌面/dataset/2069758074335653889/aligned_grounding_dino_b_roi/detections_aligned_roi_only.jsonl \
  --output-dir /tmp/roi_projection_dino_eval_smoke \
  --max-records 20 \
  --iou-threshold 0.5 \
  --low-iou-threshold 0.3
```

全量运行：

```bash
python3 compare_and_visual/evaluate_roi_projection_vs_dino.py \
  --dataset-root /home/c64508/桌面/dataset/2069758074335653889 \
  --projection-jsonl /home/c64508/桌面/dataset/2069758074335653889/vis_projection_newstruct_final_roi/projections_aligned.jsonl \
  --dino-jsonl /home/c64508/桌面/dataset/2069758074335653889/aligned_grounding_dino_b_roi/detections_aligned_roi_only.jsonl \
  --output-dir /home/c64508/桌面/dataset/2069758074335653889/compare_grounding_dino_b_roi_projection_qc \
  --iou-threshold 0.5 \
  --low-iou-threshold 0.3
```

代码检查：

```bash
python3 -m py_compile compare_and_visual/evaluate_roi_projection_vs_dino.py
```

输出检查：

```text
1. manifest.json 中输入路径、阈值、记录数正确。
2. matches_aligned.jsonl 行数等于参与评价的 aligned key 数。
3. projection_boxes 总数等于各 record projected_boxes 数量之和。
4. dino_boxes 总数等于各 record detections 数量之和。
5. geometry_tp + unmatched_projection 不应超过 projection_boxes。
6. geometry_tp + unmatched_dino 不应超过 dino_boxes。
7. 抽样检查 IoU 最高的 match 是否与 overlay 图直觉一致。
8. 抽样检查 unmatched_projection / unmatched_dino 是否确实没有合理重叠框。
```

## 15. 后续衔接

完成第一层后，第二层可以直接读取：

```text
matches_aligned.jsonl
unmatched_projection_boxes.jsonl
unmatched_dino_boxes.jsonl
metrics_summary.json
```

并映射为：

```text
unmatched_projection -> 多标错误疑似 / baseline 漏检疑似
unmatched_dino -> 漏标错误疑似 / baseline 误检疑似
class_mismatch -> 类别错误疑似
low_iou_matched / weak_overlap_matched -> 贴合错误疑似
```

注意：这些仍然是 pseudo QC 结果，需要在文档和输出字段中保留“疑似”语义。
