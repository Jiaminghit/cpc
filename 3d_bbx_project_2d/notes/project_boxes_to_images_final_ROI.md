# `project_boxes_to_images_withROI.py` ROI 版投影说明

## 1. 文档目的

本文记录当前已经完成的 ROI 版 3D 检测框投影脚本：

```text
3d_bbx_project_2d/project_boxes_to_images_withROI.py
```

该脚本基于：

```text
3d_bbx_project_2d/project_boxes_to_images_final.py
```

扩展而来。当前代码已经完成公共模块整理：`project_boxes_to_images_final.py` 和 `project_boxes_to_images_withROI.py` 都复用 `utils/` 下的检测框解析、类别映射、标定读取、几何变换、ROI 判断等公共工具；相机图像渲染逻辑统一放在 `render/camera.py` 中。两个脚本现在主要承担 CLI、index 遍历、JSONL/summary 输出等调度职责。

ROI 版脚本的核心目标是：在 3D 检测框投影阶段，基于 ego 坐标系判断目标是否位于 OD 预标质检规范中的 ROI 范围内，并将 ROI 信息写入结构化输出，供后续定量评价脚本直接使用。

质检规范中的 ROI 范围为：

```text
左右 50m，前 150m，后 50m
```

在当前数据的小样本验证中，ego 坐标轴更接近：

```text
x: front/back
y: left/right
z: up/down
```

因此脚本默认使用：

```text
lateral_axis = y
longitudinal_axis = x
lateral_range_m = [-50, 50]
longitudinal_range_m = [-50, 150]
```

## 2. 设计原理

ROI 判断必须在 3D 空间中完成，不能用 2D 图像 bbox 判断。原因是 ROI 范围来自质检规范中的车辆周围空间范围：

```text
左右 50m
前 150m
后 50m
```

当前投影脚本已经完成如下坐标链路：

```text
box source frame / lidar_top_GT
  -> ego
  -> camera
  -> pixel
```

因此，最合理的 ROI 判断位置是：

```text
3D box source frame -> ego 之后
ego -> camera 之前
```

当前实现对每个 3D box 计算：

```text
center_ego = T_box_to_ego @ box.center
```

然后根据 `center_ego` 在 ego 坐标系下的 lateral / longitudinal 值判断：

```text
in_roi =
  lateral_min <= lateral <= lateral_max
  and longitudinal_min <= longitudinal <= longitudinal_max
```

第一版实现遵循保守策略：

1. 默认只打 ROI 标记，不丢弃 ROI 外目标。
2. 只有显式传入 `--roi-filter` 时，才会在投影前过滤 ROI 外目标。
3. 每个投影框的 ROI 明细写入 `projections_aligned.jsonl`。
4. 每帧、每相机、全局 ROI 统计写入 `projection_summary.json`。
5. 后续评价脚本优先读取 `projected_boxes[].roi.in_roi`，再自行选择是否只统计 ROI 内目标。

## 3. 对外接口

### 3.1 原有参数

ROI 版脚本保留原有主要参数：

```text
--index
--output-root
--max-frames
--include-invalid
--score-threshold
--box-source-frame
--min-depth
```

含义与 `project_boxes_to_images_final.py` 一致。

### 3.2 新增 ROI 参数

当前已实现的 ROI 参数：

```text
--roi-enable
--roi-disable
--roi-filter
--save-debug-vis
--roi-lateral-axis
--roi-longitudinal-axis
--roi-lateral-min
--roi-lateral-max
--roi-longitudinal-min
--roi-longitudinal-max
```

默认值：

```text
roi_enable = true
roi_filter = false
roi_lateral_axis = y
roi_longitudinal_axis = x
roi_lateral_min = -50.0
roi_lateral_max = 50.0
roi_longitudinal_min = -50.0
roi_longitudinal_max = 150.0
```

说明：

- `--roi-enable`：启用 ROI 字段计算。该脚本默认启用。
- `--roi-disable`：关闭 ROI 字段计算。
- `--roi-filter`：启用后，ROI 外目标不进入正式 `projected_boxes` 和 `vis/` 输出。
- `--save-debug-vis`：额外输出 `debug_vis/`，显示所有可见投影框的 ROI 状态。ROI 内为绿色并标注 `roi`，ROI 外为红色并标注 `out`，ROI unknown 为灰色并标注 `unknown`。标签格式为 `Class Confidence roi/out/unknown long=Longitudinal lat=Lateral`，其中 `Longitudinal` 和 `Lateral` 分别来自 ROI 判断中的 `longitudinal_m` 和 `lateral_m`。
- `--roi-lateral-axis`：ego 坐标系中左右方向使用的轴，取值为 `x/y/z`。
- `--roi-longitudinal-axis`：ego 坐标系中前后方向使用的轴，取值为 `x/y/z`。
- `--roi-lateral-min/max`：左右方向 ROI 范围。
- `--roi-longitudinal-min/max`：前后方向 ROI 范围。

脚本会检查：

```text
roi_lateral_axis != roi_longitudinal_axis
roi_lateral_min <= roi_lateral_max
roi_longitudinal_min <= roi_longitudinal_max
```

不满足时直接报错。

## 4. 核心实现方法

当前相关代码的职责划分如下：

```text
utils/common.py          JSON 读取、路径解析、数值格式化
utils/labels.py          类别表、类别颜色、label 归一化
utils/detection.py       DetectionBox 与 prelabel-model 检测框读取
utils/calibration.py     相机标定、box source -> ego 变换
utils/geometry.py        3D box 角点、齐次变换、相机投影、bbox 计算
utils/roi.py             RoiConfig、ROI 判断、ROI 统计字段
utils/visualization.py   基础 OpenCV 画框函数
render/camera.py         统一相机图像渲染函数 render_camera_image()
```

其中 `render/camera.py` 的 `render_camera_image()` 通过可选 `roi_config` 同时支持两种调用方式：

```text
roi_config = None
  -> 行为与 project_boxes_to_images_final.py 的非 ROI 投影一致

roi_config = RoiConfig(...)
  -> 行为与 project_boxes_to_images_withROI.py 的 ROI 投影一致
```

### 4.1 `RoiConfig`

当前 `RoiConfig` 定义在：

```text
3d_bbx_project_2d/utils/roi.py
```

结构为：

```python
@dataclass(frozen=True)
class RoiConfig:
    enabled: bool
    filter_enabled: bool
    frame: str
    lateral_axis: str
    longitudinal_axis: str
    lateral_range_m: tuple[float, float]
    longitudinal_range_m: tuple[float, float]
```

该结构在 `main()` 中由 argparse 参数构造，并传入：

```text
project_index(...)
render_camera_image(...)
```

### 4.2 ROI 配置序列化

当前通过：

```python
roi_config_to_dict(roi_config)
```

将配置写入：

```text
projection_summary.json -> roi
projections_aligned.jsonl -> params.roi
```

输出示例：

```json
{
  "enabled": true,
  "filter_enabled": false,
  "frame": "ego",
  "lateral_axis": "y",
  "longitudinal_axis": "x",
  "lateral_range_m": [-50.0, 50.0],
  "longitudinal_range_m": [-50.0, 150.0]
}
```

### 4.3 ROI 逐框计算

当前通过：

```python
build_roi_info(center_ego, roi_config)
```

计算逐框 ROI 信息。

启用 ROI 时，输出结构为：

```json
{
  "enabled": true,
  "in_roi": true,
  "frame": "ego",
  "center_ego": [52.85831, -1.865252, 1.344357],
  "lateral_axis": "y",
  "longitudinal_axis": "x",
  "lateral_m": -1.865252,
  "longitudinal_m": 52.85831,
  "lateral_range_m": [-50.0, 50.0],
  "longitudinal_range_m": [-50.0, 150.0]
}
```

关闭 ROI 时，输出结构为：

```json
{
  "enabled": false,
  "in_roi": null,
  "frame": "ego",
  "center_ego": [...]
}
```

### 4.4 投影主流程

`render/camera.py -> render_camera_image()` 中的关键 ROI 流程为：

```python
corners_box = box_to_corners(box.center, box.size_lwh, box.heading)
corners_ego = transform_points(corners_box, t_box_to_ego)
center_ego = transform_points(box.center.reshape(1, 3), t_box_to_ego)[0]
roi = build_roi_info(center_ego, roi_config)
```

如果启用 `--roi-filter` 且目标不在 ROI 内，正式输出会跳过该目标。当前共享渲染函数中分两种情况处理：

1. 未开启 `--save-debug-vis` 时，ROI 外目标会在进入 camera 投影前直接跳过，并计入 `boxes_skipped_outside_roi`。
2. 开启 `--save-debug-vis` 时，ROI 外目标仍会继续完成 camera 投影和图像可见性判断，先写入 `debug_vis/`，随后再跳过正式 `projected_boxes` 和 `vis/` 输出。

核心逻辑可理解为：

```python
if roi_config.enabled and roi_config.filter_enabled and roi["in_roi"] is False:
    roi_counts["boxes_skipped_outside_roi"] += 1
    if debug_image is None:
        continue

# 如果有 debug_image，会继续完成 camera -> pixel，用于绘制 debug_vis。

if roi_config.enabled and roi_config.filter_enabled and roi["in_roi"] is False:
    continue
```

因此，`debug_vis/` 可以回查“被正式输出过滤掉、但在图像中可见”的 ROI 外目标。

否则继续正式投影输出流程：

```text
ego -> camera
camera -> pixel
生成 corners_2d
生成 bbox_xyxy
写入 projected_boxes
```

### 4.5 逐框输出字段

每个成功投影框新增：

```json
"roi": {...}
```

同时 `bbox_3d` 新增：

```json
"center_ego": [...]
```

完整示意：

```json
{
  "projection_id": "projection:1782286937435038208:camera_front_wide:0",
  "object_id": "1",
  "class_name": "Car",
  "bbox_xyxy": [1917.704581, 995.783583, 2000.381571, 1065.151925],
  "bbox_3d": {
    "center": [2.335938, 51.84375, -1.77998],
    "center_ego": [52.85831, -1.865252, 1.344357],
    "size_lwh": [4.587969, 2.016114, 1.682882],
    "heading": 1.493164
  },
  "roi": {
    "enabled": true,
    "in_roi": true,
    "frame": "ego",
    "center_ego": [52.85831, -1.865252, 1.344357],
    "lateral_axis": "y",
    "longitudinal_axis": "x",
    "lateral_m": -1.865252,
    "longitudinal_m": 52.85831,
    "lateral_range_m": [-50.0, 50.0],
    "longitudinal_range_m": [-50.0, 150.0]
  }
}
```

## 5. 输出文件变化

### 5.1 `projections_aligned.jsonl`

每条 `pcd_timestamp + camera` 记录中的 `params` 新增：

```json
"roi": {
  "enabled": true,
  "filter_enabled": false,
  "frame": "ego",
  "lateral_axis": "y",
  "longitudinal_axis": "x",
  "lateral_range_m": [-50.0, 50.0],
  "longitudinal_range_m": [-50.0, 150.0]
}
```

每个 `projected_boxes[]` 新增：

```text
roi
bbox_3d.center_ego
```

每条记录的 `projection` 新增 ROI 聚合统计：

```text
boxes_in_roi
boxes_outside_roi
boxes_projected_in_roi
boxes_projected_outside_roi
boxes_skipped_outside_roi
```

### 5.2 `debug_vis/`

如果传入：

```text
--save-debug-vis
```

输出目录新增：

```text
debug_vis/<camera>/<pcd_timestamp>.jpg
```

该目录用于检查 ROI 判断和过滤效果。它会显示所有已经成功投影到图像内的可见框：

```text
ROI 内投影框: 绿色，标签包含 roi
ROI 外投影框: 红色，标签包含 out
ROI unknown 投影框: 灰色，标签包含 unknown
```

正式 `vis/` 仍然遵循 `--roi-filter`：如果开启过滤，`vis/` 中只显示 ROI 内框；`debug_vis/` 用于回查被过滤掉但在图像中可见的 ROI 外框。

标签格式为：

```text
Class Confidence roi/out/unknown long=Longitudinal lat=Lateral
```

其中 `Longitudinal` 和 `Lateral` 使用 ROI 判断中的 ego 纵向与横向坐标，例如 `Car 0.92 roi long=52.86m lat=-1.87m`。

### 5.3 `projection_summary.json`

顶层新增 ROI 配置：

```json
"roi": {
  "enabled": true,
  "filter_enabled": false,
  "frame": "ego",
  "lateral_axis": "y",
  "longitudinal_axis": "x",
  "lateral_range_m": [-50.0, 50.0],
  "longitudinal_range_m": [-50.0, 150.0]
}
```

顶层新增全局统计：

```text
total_boxes_in_roi
total_boxes_outside_roi
total_projected_boxes_in_roi
total_projected_boxes_outside_roi
total_skipped_outside_roi
```

每帧每相机的结果中新增：

```text
boxes_in_roi
boxes_outside_roi
boxes_projected_in_roi
boxes_projected_outside_roi
boxes_skipped_outside_roi
```

## 6. 统计字段定义

当前 ROI 统计字段含义如下：

```text
boxes_in_roi
```

通过类别和分数过滤后，3D center 位于 ROI 内的 box 数量。

```text
boxes_outside_roi
```

通过类别和分数过滤后，3D center 位于 ROI 外的 box 数量。

```text
boxes_projected_in_roi
```

成功投影并位于 ROI 内的 box 数量。

```text
boxes_projected_outside_roi
```

成功投影但位于 ROI 外的 box 数量。

```text
boxes_skipped_outside_roi
```

启用 `--roi-filter` 后，因 ROI 外被跳过的 box 数量。

如果 `--roi-filter` 未启用，该字段通常为 0。

## 7. 推荐运行命令

### 7.1 小样本验证

建议先跑 1 帧，确认输出字段和 ROI 内外数量合理：

```bash
conda run -n project_task python /home/c64508/桌面/week1_convert_projection_compare/3d_bbx_project_2d/project_boxes_to_images_withROI.py \
  --index /home/c64508/桌面/dataset/2069758074335653889/aligned_index.json \
  --output-root /tmp/project_boxes_roi_smoke_default \
  --max-frames 1 \
  --include-invalid \
  --score-threshold 0.5 \
  --box-source-frame lidar_top_GT \
  --min-depth 0.1
```

当前小样本验证结果显示，默认轴向：

```text
--roi-lateral-axis y
--roi-longitudinal-axis x
```

能够将首帧前向车辆判断为 ROI 内。

### 7.2 全量运行

建议输出到新目录，避免覆盖已有 `vis_projection_newstruct_final`：

```bash
conda run -n project_task python /home/c64508/桌面/week1_convert_projection_compare/3d_bbx_project_2d/project_boxes_to_images_withROI.py \
  --index /home/c64508/桌面/dataset/2069758074335653889/aligned_index.json \
  --output-root /home/c64508/桌面/dataset/2069758074335653889/vis_projection_newstruct_final_roi \
  --max-frames 0 \
  --include-invalid \
  --score-threshold 0.5 \
  --box-source-frame lidar_top_GT \
  --min-depth 0.1
```

### 7.3 显式指定 ROI 参数

如需显式指定当前默认 ROI：

```bash
  --roi-enable \
  --roi-lateral-axis y \
  --roi-longitudinal-axis x \
  --roi-lateral-min -50 \
  --roi-lateral-max 50 \
  --roi-longitudinal-min -50 \
  --roi-longitudinal-max 150
```

### 7.4 启用 ROI 过滤

如果确认 ROI 轴向和统计无误，并希望输出中只保留 ROI 内 projected boxes：

```bash
  --roi-filter
```

不建议第一轮全量运行就启用 `--roi-filter`。第一轮最好保留全部投影框，先检查 ROI 内外数量是否合理。

### 7.5 输出 ROI debug 可视化

如果需要同时检查 ROI 内外投影框，可增加：

```bash
  --save-debug-vis
```

输出目录：

```text
<output-root>/debug_vis/<camera>/<pcd_timestamp>.jpg
```

显示规则：

```text
ROI 内可见投影框: 绿色，标签包含 roi
ROI 外可见投影框: 红色，标签包含 out
ROI unknown 可见投影框: 灰色，标签包含 unknown
```

正式 `vis/` 仍然遵循 `--roi-filter`：如果开启过滤，`vis/` 中只显示 ROI 内框；`debug_vis/` 用于回查被过滤掉但在图像中可见的 ROI 外框。

标签格式为 `Class Confidence roi/out/unknown long=Longitudinal lat=Lateral`，其中 `Longitudinal` 和 `Lateral` 使用 ROI 判断中的 ego 纵向与横向坐标。

## 8. 后续评价脚本读取方式

后续 `evaluate_projection_vs_model.py` 应优先读取：

```text
projections_aligned.jsonl
  -> projected_boxes[].roi.in_roi
```

推荐输出以下分组结果：

```text
all_objects
roi_objects_only
valid_time_match_only
valid_time_match_and_roi_only
```

其中主报告建议使用：

```text
valid_time_match_and_roi_only
```

评价脚本中不需要重新计算 ego 坐标，也不需要重新做 box source frame -> ego 的变换，直接读取：

```text
projected_boxes[].bbox_3d.center_ego
projected_boxes[].roi.in_roi
projected_boxes[].roi.lateral_m
projected_boxes[].roi.longitudinal_m
```

即可进行 ROI 内外筛选和质检统计。

## 9. 当前验证结果

已完成以下验证：

```bash
python -m compileall 3d_bbx_project_2d
conda run -n project_task python 3d_bbx_project_2d/project_boxes_to_images_final.py --help
conda run -n project_task python 3d_bbx_project_2d/project_boxes_to_images_withROI.py --help
```

通过。

使用 `project_task` 环境运行 1 帧小样本，`project_boxes_to_images_withROI.py` 默认 ROI 模式结果为：

```text
output_root = /tmp/project_boxes_roi_smoke_default
frame_count = 1
image_count = 6
total_projected_boxes = 2
total_projected_boxes_in_roi = 2
total_projected_boxes_outside_roi = 0
```

同时验证过以下分支：

```text
project_boxes_to_images_final.py
  -> 非 ROI 调用 render_camera_image()，生成 6 路相机投影图

project_boxes_to_images_withROI.py --roi-disable
  -> 保留 ROI 版输出结构，但 ROI 统计为 0

project_boxes_to_images_withROI.py --save-debug-vis --roi-filter
  -> 生成 6 路正式投影图和 6 路 debug_vis 图
```

抽查首个投影框：

```json
{
  "center_ego": [52.85831, -1.865252, 1.344357],
  "roi": {
    "in_roi": true,
    "lateral_axis": "y",
    "longitudinal_axis": "x",
    "lateral_m": -1.865252,
    "longitudinal_m": 52.85831
  }
}
```

说明当前默认轴向对该数据样本是合理的：

```text
x: front/back
y: left/right
```

## 10. 后续版本更新策略

### 10.1 第一版已经完成

当前第一版已经完成：

- 新增独立脚本 `project_boxes_to_images_withROI.py`。
- `project_boxes_to_images_final.py` 和 `project_boxes_to_images_withROI.py` 均复用 `utils/` 公共工具。
- 相机渲染逻辑已抽到 `render/camera.py`，并通过可选 `roi_config` 同时支持非 ROI 和 ROI 调用。
- 默认计算逐框 ROI 字段。
- 默认不丢弃 ROI 外目标。
- 支持 `--roi-filter` 显式过滤 ROI 外目标。
- 输出 `projection_summary.json` 顶层 ROI 配置。
- 输出每相机和全局 ROI 统计。
- 在 `bbox_3d` 中增加 `center_ego`。

### 10.2 下一版建议

后续可以按以下顺序继续扩展：

1. 全量运行 ROI 版投影，生成 `vis_projection_newstruct_final_roi`。
2. 检查全量 ROI 内外数量分布，确认默认轴向是否稳定合理。
3. 编写 `compare_and_visual/evaluate_projection_vs_model.py`，优先读取 `projected_boxes[].roi.in_roi`。
4. 在评价脚本中输出：
   - `all_objects`
   - `roi_objects_only`
   - `valid_time_match_only`
   - `valid_time_match_and_roi_only`
5. 引入 IoU、中心点距离、面积比例、覆盖比例等 baseline consistency 指标。
6. 再引入 OD 质检错误类型映射，例如多标疑似、漏标疑似、类别错疑似、贴合差疑似。

### 10.3 暂不在投影脚本中实现的内容

以下内容暂不放在 `project_boxes_to_images_withROI.py` 中：

- 遮挡程度判断。
- 高优 / 低优错误判断。
- GroundingDINO prompt 调整和重新检测。
- OD 9 类完整类别扩展。
- IoU / TP / FP / FN / pseudo metrics。
- 质检错误类型判定。

这些逻辑更适合放在后续评价脚本中，避免投影脚本承担过多评估职责。

## 11. 注意事项

1. ROI 判断基于 3D box center，而不是 3D box corners。
2. 当前默认轴向来自小样本验证，正式报告前仍建议抽查更多样本。
3. `--roi-filter` 会改变 `projected_boxes` 内容，建议只有在确认 ROI 轴向无误后再启用。
4. 若后续数据集 ego 坐标定义不同，应通过参数调整 `--roi-lateral-axis` 和 `--roi-longitudinal-axis`。
5. 后续评价主流程建议使用未开启 `--roi-filter` 的输出，再由评价脚本按 `roi.in_roi` 做筛选，这样更利于回溯。
