# `project_boxes_to_bev.py` BEV 可视化实现计划

## 1. 任务目标

在 `3d_bbx_project_2d/` 下新增脚本：

```text
project_boxes_to_bev.py
```

用于将数据集目录 `pcd/prelabel-model/` 中的 3D 目标检测框投影到 BEV 视角并输出可视化图片与结构化结果。运行环境使用现有虚拟环境：

```text
project_task
```

典型输入示例：

```text
/home/c64508/桌面/dataset/2069758074335653889/pcd/prelabel-model
```

优先支持通过已有 `aligned_index.json` 读取每帧的检测文件和标定文件；同时保留直接从 `pcd/meta.json` / `pcd/prelabel-model` 扫描的能力，便于只做 BEV 而不依赖相机图像。

## 2. 当前公共模块状态

当前 `3d_bbx_project_2d/` 已经完成公共工具与渲染模块拆分，后续 `project_boxes_to_bev.py` 不需要再从 `project_boxes_to_images_withROI.py` 或 `project_boxes_to_images_final.py` 中复制函数，而是直接复用这些模块：

```text
3d_bbx_project_2d/utils/
  __init__.py
  common.py
  labels.py
  detection.py
  calibration.py
  geometry.py
  roi.py
  visualization.py

3d_bbx_project_2d/render/
  __init__.py
  camera.py
```

其中：

- `utils/` 负责数据读取、类别映射、检测框解析、标定读取、几何变换、ROI 判断和基础绘制。
- `render/camera.py` 负责相机图像上的 3D box 渲染，当前通过可选 `roi_config` 同时支持非 ROI 版和 ROI 版相机投影。
- `project_boxes_to_images_final.py` 和 `project_boxes_to_images_withROI.py` 已经改为复用 `utils/` 与 `render/camera.py`，脚本本身主要保留 CLI、index 遍历、输出 JSONL/summary 的调度逻辑。

BEV 脚本实现时应优先复用的内容：

- `DetectionBox`：统一 3D 检测框结构。
- `RoiConfig`：复用 ROI 配置结构。
- `load_json()`：读取 JSON。
- `resolve_path()`：解析 index 中的相对路径。
- `load_detection_boxes()`：从 `prelabel-model/*.json` 读取检测框，并应用 score threshold。
- `canonicalize_label()`：类别别名归一化，复用 `Car / Pedestrian / Cyclist / Van / Traffic_cone` 类别体系。
- `box_to_corners()`：由中心点、长宽高、heading 计算 8 个 3D 角点。
- `transform_points()`：执行 4x4 齐次变换。
- `load_box_source_transform()`：读取 `box_source_frame -> ego` 变换。
- `build_roi_info()` / `roi_config_to_dict()` / `empty_roi_counts()`：复用 ROI 判断与统计逻辑。
- `CLASS_COLORS` / `DEFAULT_COLOR`：类别颜色保持和现有 2D 投影脚本一致。

当前模块归属：

```text
utils/common.py
  load_json()
  resolve_path()
  round_float()
  array_to_rounded_list()

utils/labels.py
  CLASS_TABLE
  CLASS_ID_BY_NAME
  ALIASES
  CLASS_PRIORITY
  CLASS_COLORS
  DEFAULT_COLOR
  canonicalize_label()

utils/detection.py
  DetectionBox
  load_detection_boxes()

utils/calibration.py
  find_sensor()
  sensor_transform_matrix()
  intrinsic_camera_matrix()
  load_box_source_transform()
  load_calibration()

utils/geometry.py
  BOX_EDGES
  box_to_corners()
  transform_points()
  project_points()
  bbox_xyxy_from_points()
  clip_bbox_xyxy()
  projected_bbox_intersects()

utils/roi.py
  RoiConfig
  ROI_FRAME
  ROI_AXIS_INDEX
  roi_config_to_dict()
  empty_roi_counts()
  build_roi_info()

utils/visualization.py
  draw_box_edges()

render/camera.py
  render_camera_image()
```

注意：`3d_bbx_project_2d` 目录名以数字开头，不建议把它作为常规 Python 包名导入。更合适的方式是在该目录内的脚本中使用：

```python
from utils.detection import DetectionBox, load_detection_boxes
from utils.geometry import box_to_corners, transform_points
from utils.roi import RoiConfig, build_roi_info
```

执行 `python 3d_bbx_project_2d/project_boxes_to_bev.py ...` 时，Python 会把脚本所在目录放入 `sys.path`，因此上述 `utils.*` 导入可直接工作。

如果后续希望把 BEV 绘制也独立出来，建议新增：

```text
3d_bbx_project_2d/render/bev.py
```

用于承载 `build_bev_canvas()`、`ego_to_bev_pixels()`、`render_bev_frame()` 等 BEV 专属绘制逻辑，使 `project_boxes_to_bev.py` 保持和现有相机投影脚本类似的调度层定位。

## 3. 输入与坐标约定

检测框 JSON 结构为 GeoJSON 风格：

```text
features[*].geometry.coordinates[0] = center
features[*].geometry.coordinates[1] = size_lwh
features[*].geometry.coordinates[2][2] = heading
features[*].properties.type = class label
features[*].properties.score = confidence
```

BEV 绘制使用 ego 平面坐标：

```text
x: longitudinal / front-back
y: lateral / left-right
z: up-down
```

默认 ROI 范围沿用现有 ROI 脚本：

```text
lateral_axis = y
longitudinal_axis = x
lateral_range_m = [-50, 50]
longitudinal_range_m = [-50, 150]
```

`prelabel-model` 的 metadata 中可能声明 `crs = Ego`，但现有相机投影脚本默认 `box_source_frame = lidar_top_GT`。为兼容两种情况，BEV 脚本需要提供：

```text
--box-source-frame lidar_top_GT
--box-source-frame ego
```

当 `--box-source-frame ego` 时不需要标定变换；否则从当前帧 calibration 中读取对应 sensor transform，得到 `T_box_source_to_ego`。

## 4. CLI 参数设计

建议参数：

```text
--index
--dataset-root
--prelabel-dir
--output-root
--max-frames
--include-invalid
--score-threshold
--box-source-frame
--bev-width
--bev-height
--meters-per-pixel
--x-min
--x-max
--y-min
--y-max
--draw-labels / --no-draw-labels
--draw-heading / --no-draw-heading
--roi-enable
--roi-disable
--roi-filter
--roi-lateral-axis
--roi-longitudinal-axis
--roi-lateral-min
--roi-lateral-max
--roi-longitudinal-min
--roi-longitudinal-max
```

默认值建议：

```text
--index /home/c64508/桌面/dataset/2069758074335653889/aligned_index.json
--output-root <dataset_root>/vis_bev_prelabel_model
--score-threshold 0.0
--box-source-frame lidar_top_GT
--x-min -60
--x-max 160
--y-min -60
--y-max 60
--meters-per-pixel 0.2
--bev-width / --bev-height 默认由范围和 meters-per-pixel 自动计算
--roi-enable 默认开启
--roi-filter 默认关闭
```

其中 `--roi-filter` 是本任务要求预留的 ROI 过滤开关：默认只计算和记录 ROI 状态，不丢弃 ROI 外目标；只有显式传入时才过滤 ROI 外目标。

## 5. BEV 绘制方法

### 5.1 坐标映射

将 ego 坐标中的点映射到图像像素：

```text
u = (y - y_min) / meters_per_pixel
v = (x_max - x) / meters_per_pixel
```

这样图像上方表示前方，图像右侧表示车辆右/左需要根据数据轴方向最终确认。若发现左右方向反了，可新增 `--flip-y` 或通过交换 `y_min/y_max` 处理。

### 5.2 绘制元素

每帧 BEV 输出一张图，建议包含：

- 深色或浅色背景。
- ego 原点和自车朝向箭头。
- 10m 或 20m 网格线。
- ROI 边界矩形，默认范围为左右 50m、前 150m、后 50m。
- 每个 3D box 的底面四边形。
- heading 方向短箭头。
- 类别 + score 标签，可通过 `--no-draw-labels` 关闭。

3D box 的 BEV 多边形使用 `box_to_corners()` 生成 8 个角点后，取底面角点或直接取 XY 平面四个角点绘制。为避免高度方向影响 BEV，绘制时只使用 `corners_ego[:, [x, y]]`。

### 5.3 ROI 可视化

当 ROI 启用时：

- ROI 内目标使用类别颜色正常绘制。
- ROI 外目标在未开启 `--roi-filter` 时仍绘制，但可降低亮度或使用灰色虚线/细线。
- 开启 `--roi-filter` 时，ROI 外目标不进入正式 BEV 输出和结构化 `boxes` 列表。
- 可在 summary 中统计 `boxes_in_roi`、`boxes_outside_roi`、`boxes_projected_in_roi`、`boxes_skipped_outside_roi`。

## 6. 输出结构

建议输出目录：

```text
<output-root>/
  aligned_index.json
  bev_summary.json
  bev_boxes.jsonl
  bev/
    <pcd_timestamp>.jpg
```

`bev_boxes.jsonl` 每行对应一个 PCD frame：

```json
{
  "schema_version": "bev_projection.v1",
  "dataset_id": "2069758074335653889",
  "pcd_timestamp": 1782286937435038208,
  "det_json": ".../pcd/prelabel-model/1782286937435038208.json",
  "bev_image": ".../bev/1782286937435038208.jpg",
  "params": {
    "score_threshold": 0.0,
    "box_source_frame": "lidar_top_GT",
    "meters_per_pixel": 0.2,
    "x_range_m": [-60, 160],
    "y_range_m": [-60, 60],
    "roi": {}
  },
  "boxes": []
}
```

每个 `boxes[]` 建议包含：

- `object_id`
- `class_name`
- `class_id`
- `label_raw`
- `score`
- `bbox_3d.center`
- `bbox_3d.center_ego`
- `bbox_3d.size_lwh`
- `bbox_3d.heading`
- `bev_polygon_px`
- `bev_polygon_ego`
- `roi`
- `visibility.projected`
- `visibility.skip_reason`

`bev_summary.json` 汇总：

- 输入 dataset / index / output。
- 处理帧数。
- 总 box 数。
- 已绘制 box 数。
- 未知类别数量。
- ROI 内外与过滤数量。
- BEV 画布参数。

## 7. 实现步骤

1. 新建 `3d_bbx_project_2d/project_boxes_to_bev.py`。
2. 从 `utils/` 复用检测框解析、类别映射、标定读取、坐标变换、ROI 配置与 ROI 判断。
3. 可选新增 `3d_bbx_project_2d/render/bev.py`，将 BEV 画布、坐标映射和帧绘制逻辑放入渲染层。
4. 实现 `parse_args()`，覆盖 index、dataset、输出、BEV 范围、score threshold、ROI 开关。
5. 实现 frame 输入收集：
   - 优先从 `aligned_index.json` 读取 `frames[*].det_json` 和 `frames[*].calib.path`。
   - 当未提供 index 时，从 `dataset_root/pcd/meta.json` 读取 `preann.prelabel-model`。
6. 实现 `build_bev_canvas()`：
   - 根据范围和分辨率创建画布。
   - 绘制网格、ego 原点、自车箭头、ROI 边界。
7. 实现 `ego_to_bev_pixels()` 坐标映射函数。
8. 实现 `render_bev_frame()`：
   - 加载检测框。
   - 归一化类别。
   - 计算 box 角点。
   - 转换到 ego 坐标。
   - 计算 ROI。
   - 根据 `--roi-filter` 决定是否跳过。
   - 绘制 BEV 多边形、heading、label。
   - 返回当前帧结构化结果和统计。
9. 实现主循环与输出：
   - 保存 `bev/<timestamp>.jpg`。
   - 写入 `bev_boxes.jsonl`。
   - 写入 `bev_summary.json`。
   - 若从 index 输入，复制 `aligned_index.json` 到输出目录，方便结果追溯。
10. 使用 `project_task` 环境运行小样本验证。
11. 检查输出图片是否非空、ROI 边界位置是否合理、summary 统计是否与 JSONL 汇总一致。

## 8. 验证命令建议

小样本验证：

```bash
conda run -n project_task python 3d_bbx_project_2d/project_boxes_to_bev.py \
  --index /home/c64508/桌面/dataset/2069758074335653889/aligned_index.json \
  --max-frames 5 \
  --output-root /home/c64508/桌面/dataset/2069758074335653889/vis_bev_prelabel_model_test
```

ROI 过滤验证：

```bash
conda run -n project_task python 3d_bbx_project_2d/project_boxes_to_bev.py \
  --index /home/c64508/桌面/dataset/2069758074335653889/aligned_index.json \
  --max-frames 5 \
  --roi-filter \
  --output-root /home/c64508/桌面/dataset/2069758074335653889/vis_bev_prelabel_model_test_roi
```

如果确认检测框已经是 ego 坐标，可额外验证：

```bash
conda run -n project_task python 3d_bbx_project_2d/project_boxes_to_bev.py \
  --index /home/c64508/桌面/dataset/2069758074335653889/aligned_index.json \
  --box-source-frame ego \
  --max-frames 5 \
  --output-root /home/c64508/桌面/dataset/2069758074335653889/vis_bev_prelabel_model_test_ego
```

## 9. 风险与待确认点

- `prelabel-model` JSON metadata 声明 `crs = Ego`，但现有相机投影脚本默认使用 `lidar_top_GT -> ego`。实现时需保留 `--box-source-frame`，并通过小样本图像检查确认默认值是否需要改成 `ego`。
- BEV 图像中左右方向可能需要根据业务习惯调整，必要时新增 `--flip-lateral-axis`。
- 部分类别可能无法被 `canonicalize_label()` 识别，应在 summary 中记录未知类别数量，避免静默丢失。
- 如果只提供 `prelabel-dir` 且 `--box-source-frame` 不是 `ego`，脚本无法获得标定文件，需要报错提示改用 `--index` 或 `--dataset-root`。
- ROI 判断默认基于 box center，和现有 ROI 投影脚本保持一致；如果后续要求“框任意部分进入 ROI 即保留”，需要新增 `--roi-mode center/intersects`。
