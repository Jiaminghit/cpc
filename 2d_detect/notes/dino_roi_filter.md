# DINO 检测框 LiDAR ROI 过滤实现方案

## 1. 目标

本文档用于指导后续新增脚本：

```text
2d_detect/filter_dino_detections_by_lidar_roi.py
```

该脚本的目标是：在已经得到按 `pcd_timestamp + camera` 对齐的 GroundingDINO 2D 检测结果后，利用同一帧 LiDAR 点云和相机标定，估计每个 DINO 2D 检测框在 ego 坐标系下的位置，并据此判断该检测框是否位于 OD 质检 ROI 范围内。

核心链路为：

```text
LiDAR PCD 点云
  -> lidar/source frame
  -> ego frame
  -> camera frame
  -> image pixel
  -> 落入 DINO bbox 的点云点
  -> 估计 bbox 的 ego 位置
  -> 判断 ROI
  -> 输出 ROI 增强版 DINO 检测结果
```

需要强调：DINO 2D bbox 本身没有深度，不能直接反投影成唯一 3D 位置。第一版实现应采用“点云投影到图像，再与 2D bbox 关联”的方式。

## 2. 输入文件

以当前数据集为例：

```text
/home/c64508/桌面/dataset/2069758074335653889
```

脚本建议输入：

```text
--dataset-root
--aligned-index
--detections-jsonl
--output-dir
--box-source-frame
--min-depth
--roi-lateral-axis
--roi-longitudinal-axis
--roi-lateral-min
--roi-lateral-max
--roi-longitudinal-min
--roi-longitudinal-max
```

默认路径建议：

```text
aligned_index:
  <dataset-root>/aligned_index.json

detections-jsonl:
  <dataset-root>/aligned_grounding_dino_b/detections_aligned.jsonl

output-dir:
  <dataset-root>/aligned_grounding_dino_b_roi
```

关键输入说明：

```text
aligned_index.json
```

提供每个 `pcd_timestamp` 对应的：

- PCD 点云路径。
- 标定 JSON 路径。
- 每路相机的 RGB 图像路径和时间对齐信息。

```text
detections_aligned.jsonl
```

提供每个 `pcd_timestamp + camera` 下的 DINO 检测结果：

- `bbox_xyxy`
- `bbox_cxcywh_norm`
- `class_name`
- `score`
- `rgb_image`
- `image_width`
- `image_height`

```text
struct_json/calib/*.json
```

提供 LiDAR 和相机外参，以及相机内参和畸变参数。

```text
pcd/p128_0/*.pcd
```

提供当前 `pcd_timestamp` 对应的 LiDAR 点云。实际读取时不应硬编码遍历 `pcd/` 根目录，而应优先读取 `aligned_index.json` 中的：

```text
frame["pcd"]["path"]
```

在当前数据集中，该字段通常指向：

```text
pcd/p128_0/<pcd_timestamp>.pcd
```

## 3. 输出文件

建议输出目录结构：

```text
<dataset-root>/aligned_grounding_dino_b_roi/
  manifest.json
  detections_aligned_with_roi.jsonl
  detections_aligned_roi_only.jsonl
  roi_filter_summary.json
  vis/
    <camera>/<pcd_timestamp>.jpg
  vis_by_image/
    <camera>/<rgb_timestamp>.jpg
  debug_vis/
    <camera>/<pcd_timestamp>.jpg
```

其中：

- `vis/`：保存 ROI 过滤后的 DINO 可视化结果，按 `pcd_timestamp + camera` 对齐，是后续和 3D 投影结果对比的主可视化目录。
- `vis_by_image/`：保存 ROI 过滤后的 DINO 可视化结果，按原始 RGB 图像时间戳保存，便于从图片本身回查。
- `debug_vis/`：可选调试目录，额外绘制 LiDAR 投影点、bbox 内点和代表点，用于验证 ROI 过滤是否可信。

`vis/` 和 `vis_by_image/` 应尽量与原始 `aligned_grounding_dino_b/` 目录保持同构。这样过滤前和过滤后的 DINO 结果可以直接并排检查。

### 3.1 `detections_aligned_with_roi.jsonl`

每行仍对应一个：

```text
pcd_timestamp + camera
```

该文件保留原始 DINO 检测框，但在每个 detection 中新增：

```json
{
  "lidar_roi": {
    "enabled": true,
    "in_roi": true,
    "reason": null,
    "frame": "ego",
    "center_ego": [52.3, -1.8, 0.6],
    "representative_point_ego": [52.3, -1.8, 0.6],
    "representative_point_camera": [1.2, 0.4, 52.0],
    "depth_camera_m": 52.0,
    "points_in_bbox": 38,
    "points_used": 12,
    "lateral_axis": "y",
    "longitudinal_axis": "x",
    "lateral_m": -1.8,
    "longitudinal_m": 52.3,
    "lateral_range_m": [-50.0, 50.0],
    "longitudinal_range_m": [-50.0, 150.0],
    "method": "project_lidar_points_to_image_bbox_depth_percentile"
  }
}
```

如果无法可靠估计位置，则输出：

```json
{
  "lidar_roi": {
    "enabled": true,
    "in_roi": null,
    "reason": "insufficient_lidar_points",
    "frame": "ego",
    "center_ego": null,
    "points_in_bbox": 0,
    "points_used": 0
  }
}
```

### 3.2 `detections_aligned_roi_only.jsonl`

每行仍对应一个 `pcd_timestamp + camera`，但 `detections` 只保留：

```text
detection.lidar_roi.in_roi == true
```

`in_roi == false` 和 `in_roi == null` 默认不进入 ROI-only 主评价文件。

### 3.3 `roi_filter_summary.json`

记录整体统计：

```text
num_records
num_detections_total
num_detections_in_roi
num_detections_outside_roi
num_detections_unknown_roi
num_records_without_pcd
num_records_without_calib
num_records_without_projected_points
```

并按 camera、class 拆分统计。

### 3.4 `vis/` 和 `vis_by_image/`

如果传入：

```text
--save-vis
```

脚本应输出 ROI 过滤后的 DINO 检测框可视化结果。

```text
vis/<camera>/<pcd_timestamp>.jpg
```

用于按主对齐键查看 ROI 过滤后的 DINO baseline。该目录应作为后续与 3D 投影可视化结果对比的主目录。

```text
vis_by_image/<camera>/<rgb_timestamp>.jpg
```

用于按原始 RGB 图像查看 ROI 过滤后的 DINO baseline。若同一张 RGB 图被多个 PCD frame 复用，该目录可能发生同名覆盖，因此严谨对齐仍以 `vis/` 和 JSONL 为准。

建议可视化颜色：

```text
正式 vis / vis_by_image:
  只显示 ROI 内 detection
  按 class 使用不同颜色

debug_vis:
  ROI 内 detection: 绿色
  ROI 外 detection: 红色
  ROI unknown detection: 灰色
```

正式 ROI-only 可视化不显示 ROI 外和 ROI unknown detection，避免和后续评价输入不一致。

`debug_vis` 中 detection 标签格式为：

```text
Class Confidence roi/out/unknown Distance
```

其中 `Distance` 使用 `lidar_roi.longitudinal_m`，也就是 ROI 判断中的 ego 纵向坐标。例如：

```text
Car 0.69 roi 50.65m
```

### 3.5 `debug_vis/`

如果传入：

```text
--save-debug-vis
```

脚本应额外输出调试可视化：

```text
debug_vis/<camera>/<pcd_timestamp>.jpg
```

该目录用于检查 LiDAR 点云投影与 DINO bbox 的几何关系，建议画出：

- 原始 DINO bbox。
- bbox 内所有 LiDAR 投影点。
- 用于估计代表 ego 位置的 LiDAR 点。
- detection 的 ROI 状态。

## 4. 坐标变换链路

后续脚本应尽量复用 `3d_bbx_project_2d/project_boxes_to_images_withROI.py` 中的标定读取和点变换逻辑。

当前 3D 投影脚本使用的链路为：

```text
box source frame / lidar_top_GT
  -> ego
  -> camera
  -> pixel
```

DINO ROI 过滤脚本中的点云链路应保持一致：

```text
pcd points in lidar/source frame
  -> T_lidar_to_ego
  -> points_ego
  -> T_ego_to_camera
  -> points_camera
  -> camera intrinsics/distortion
  -> points_pixel
```

默认 `--box-source-frame` 建议沿用：

```text
lidar_top_GT
```

如果 PCD 点云实际坐标系与 `lidar_top_GT` 不一致，应后续通过小样本可视化验证并调整。

## 5. ROI 定义

沿用 `project_boxes_to_images_withROI.py` 默认 ROI 参数：

```text
frame = ego
lateral_axis = y
longitudinal_axis = x
lateral_range_m = [-50, 50]
longitudinal_range_m = [-50, 150]
```

即：

```text
lateral_min <= center_ego[lateral_axis] <= lateral_max
and
longitudinal_min <= center_ego[longitudinal_axis] <= longitudinal_max
```

第一版只判断中心代表点是否在 ROI 内，不对 2D bbox 面积或点云簇范围做复杂判断。

## 6. 实现步骤

### 6.1 读取对齐索引

读取 `aligned_index.json`，构建两个 lookup：

```text
frame_by_pcd_timestamp[pcd_timestamp] = frame
record_key = (pcd_timestamp, camera)
```

每个 frame 中需要使用：

```text
frame["pcd"]["path"]
frame["calib"]["path"]
frame["images"][camera]
```

### 6.2 读取 DINO 检测结果

逐行读取 `detections_aligned.jsonl`。

每行使用：

```text
pcd_timestamp
camera
image_width
image_height
detections[]
```

脚本应保留原始记录结构，只在 detection 内新增 `lidar_roi` 字段。

### 6.3 加载标定

根据当前记录的 `pcd_timestamp` 找到 frame，再读取：

```text
frame["calib"]["path"]
```

需要得到：

```text
T_lidar_to_ego
T_camera_to_ego
T_ego_to_camera = inverse(T_camera_to_ego)
camera_matrix
distortion_coeffs
distortion_model
image_width
image_height
```

### 6.4 加载 PCD 点云

根据当前 frame 读取：

```text
frame["pcd"]["path"]
```

第一版可优先支持 ASCII 或常见 PCD 格式。如果环境中已有 `open3d`，可以用 `open3d.io.read_point_cloud`；如果没有，则实现一个轻量 PCD reader，读取 `FIELDS x y z` 的点。

建议输出点云数组：

```text
points_lidar: np.ndarray, shape = [N, 3]
```

缓存策略：

```text
pcd_cache[pcd_timestamp] = projected_points_by_camera
```

由于同一个 PCD frame 会对应 6 路相机，建议不要为每个 detection 重复加载点云。

### 6.5 点云投影到图像

对每个 `pcd_timestamp + camera`：

```text
points_ego = transform_points(points_lidar, T_lidar_to_ego)
points_camera = transform_points(points_ego, T_ego_to_camera)
```

过滤：

```text
points_camera[:, 2] > min_depth
```

再使用相机模型投影：

```text
points_pixel = project_points(points_camera, calibration)
```

继续过滤图像范围：

```text
0 <= x < image_width
0 <= y < image_height
```

最终保留同步数组：

```text
visible_points_pixel
visible_points_camera
visible_points_ego
```

### 6.6 为每个 DINO bbox 筛选点云点

对 detection 的：

```text
bbox_xyxy = [x1, y1, x2, y2]
```

筛选：

```text
x1 <= point_pixel_x <= x2
y1 <= point_pixel_y <= y2
```

得到：

```text
points_in_bbox_pixel
points_in_bbox_camera
points_in_bbox_ego
```

如果点数小于阈值：

```text
points_in_bbox < min-points-in-bbox
```

则：

```text
lidar_roi.in_roi = null
lidar_roi.reason = "insufficient_lidar_points"
```

默认阈值建议：

```text
min_points_in_bbox = 3
```

### 6.7 估计 DINO bbox 的代表 3D 位置

第一版建议使用稳健但简单的方法：

```text
1. 取 bbox 内所有投影点作为 debug/统计点。
2. 按 bbox 垂直方向的相对范围裁剪候选点，默认使用 `y_ratio = [0.0, 0.85]`，避免 bbox 下沿地面点主导深度估计。
3. 对候选点取 camera depth = points_camera[:, 2]。
4. 按 depth 从小到大排序。
5. 取最近的 depth_percentile 点，例如前 20%。
6. 如果前 20% 点数少于 min_points_used，则至少取 min_points_used 个。
7. 对选中的 points_ego 取 median，作为 representative_point_ego。
```

默认参数建议：

```text
depth_percentile = 0.2
min_points_used = 3
bbox_candidate_y_min_ratio = 0.0
bbox_candidate_y_max_ratio = 0.85
```

这样做的原因：

- 2D bbox 内可能包含背景点。
- 背景点通常更远。
- bbox 下沿可能包含近处地面点，尤其远处目标会被透视投影放大这一问题。
- 先排除 bbox 底部一段，再取较近的一簇点，更可能对应 DINO 检测到的前景目标。
- median 比 mean 更抗离群点。

后续可扩展为深度聚类或 DBSCAN，但第一版不需要复杂化。

### 6.8 ROI 判断

用代表点：

```text
center_ego = representative_point_ego
```

计算：

```text
lateral_m = center_ego[lateral_axis]
longitudinal_m = center_ego[longitudinal_axis]
in_roi = lateral_min <= lateral_m <= lateral_max
         and longitudinal_min <= longitudinal_m <= longitudinal_max
```

写入 detection：

```text
detection["lidar_roi"] = roi_info
```

### 6.9 写出结果

对每条原始 DINO aligned record：

1. 写入完整增强版到 `detections_aligned_with_roi.jsonl`。
2. 复制一份 record，将 `detections` 过滤为 `lidar_roi.in_roi == true`，写入 `detections_aligned_roi_only.jsonl`。
3. 累加 summary 统计。

## 7. 命令行参数建议

建议第一版 CLI：

```bash
python 2d_detect/filter_dino_detections_by_lidar_roi.py \
  --dataset-root /home/c64508/桌面/dataset/2069758074335653889 \
  --aligned-index /home/c64508/桌面/dataset/2069758074335653889/aligned_index.json \
  --detections-jsonl /home/c64508/桌面/dataset/2069758074335653889/aligned_grounding_dino_b/detections_aligned.jsonl \
  --output-dir /home/c64508/桌面/dataset/2069758074335653889/aligned_grounding_dino_b_roi \
  --box-source-frame lidar_top_GT \
  --min-depth 0.1 \
  --min-points-in-bbox 3 \
  --depth-percentile 0.2 \
  --bbox-candidate-y-min-ratio 0.0 \
  --bbox-candidate-y-max-ratio 0.85 \
  --roi-lateral-axis y \
  --roi-longitudinal-axis x \
  --roi-lateral-min -50 \
  --roi-lateral-max 50 \
  --roi-longitudinal-min -50 \
  --roi-longitudinal-max 150
```

可选调试参数：

```text
--max-records
--cameras
--valid-time-only
--save-vis
--save-debug-vis
```

## 8. Debug 可视化建议

第一版建议同时支持正式可视化和 debug 可视化。

正式 ROI 过滤结果可视化：

```text
--save-vis
```

输出目录：

```text
<output-dir>/vis/<camera>/<pcd_timestamp>.jpg
<output-dir>/vis_by_image/<camera>/<rgb_timestamp>.jpg
```

正式可视化主要用于查看 ROI 过滤后的 DINO baseline，不一定需要画 LiDAR 点。

debug 可视化：

```text
--save-debug-vis
```

输出目录：

```text
<output-dir>/debug_vis/<camera>/<pcd_timestamp>.jpg
```

建议画：

- 原始 DINO bbox。
- bbox 内被选中的 LiDAR 投影点。
- 用于估计代表位置的点用另一种颜色。
- ROI 内 bbox 和 ROI 外 bbox 用不同颜色。
- `unknown_roi` 用灰色。
- bbox 标签显示 `Class Confidence roi/out/unknown Distance`，其中 `Distance` 使用 `lidar_roi.longitudinal_m`。

这一步对于验证坐标链路非常重要。如果点云投影整体错位，不能继续相信 ROI 结果。

## 9. 质量检查和验证顺序

建议按以下顺序验证：

1. 只跑 1 个 `pcd_timestamp + camera_front_wide`。
2. 确认点云投影点落在图像中合理位置。
3. 抽查若干 DINO bbox 内是否能找到点。
4. 确认近处车辆的 `longitudinal_m`、`lateral_m` 数值合理。
5. 确认 ROI 内外判断符合直觉。
6. 再跑 1 帧 6 路相机。
7. 最后跑全量。

建议 summary 中重点看：

```text
unknown_roi ratio
outside_roi ratio
points_in_bbox median
points_used median
```

如果 `unknown_roi` 比例过高，优先检查：

- PCD 是否读取正确。
- LiDAR source frame 是否正确。
- 标定外参方向是否正确。
- 相机畸变模型是否正确。
- RGB 与 PCD 时间差是否过大。
- DINO bbox 是否集中在远距离或小目标上。

## 10. 与后续评价脚本的关系

后续 `evaluate_projection_vs_model.py` 应优先读取：

```text
3D 投影结果:
  vis_projection_newstruct_final_roi/projections_aligned.jsonl

DINO ROI 结果:
  aligned_grounding_dino_b_roi/detections_aligned_roi_only.jsonl
```

评价主键保持：

```text
pcd_timestamp + camera
```

评价对象：

```text
reference boxes:
  projected_boxes[] 且 projected_boxes[].roi.in_roi == true

baseline boxes:
  detections[]，来自 detections_aligned_roi_only.jsonl
```

这样可以实现：

```text
ROI 内 3D 检测框投影
vs
ROI 内 GroundingDINO 2D baseline
```

## 11. 第一版不做的事情

为了控制实现复杂度，第一版不建议做：

- 根据 DINO bbox 生成完整 3D box。
- 对 bbox 内点云做复杂实例分割。
- 估计目标真实尺寸、朝向或 3D IoU。
- 用 DINO 检测直接修正 3D 投影框。
- 把 `in_roi == null` 强行归为 ROI 内或 ROI 外。

第一版只负责为 DINO 2D detection 增加一个可解释的 LiDAR-based ROI 字段，并输出 ROI-only baseline。
