# 3D 投影结果与模型检测结果对齐对比方案

## 目标

以单个数据集目录为单位，例如：

```text
/home/c64508/桌面/dataset/2069758074335653889
```

在已有 `aligned_index.json` 的基础上，把 3D 检测框投影到对应 RGB 图像，并额外输出结构化的 2D 投影框结果。这样后续可以把这些投影框与 GroundingDINO、YOLO、SAM 等模型检测结果按同一个键对齐：

```text
dataset_id + pcd_timestamp + camera
```

最终目标是支持：

```text
3D 检测框
  -> 投影到对齐 RGB 图像
  -> 输出投影可视化图
  -> 输出结构化 2D 投影框 JSONL
  -> 与模型检测 detections_aligned.jsonl 对齐
  -> 生成对比可视化图和数值指标
```

## 当前进度

已经完成：

```text
第一阶段：投影输出结构修改
  -> 新增 project_boxes_to_images_final.py
  -> 输出 projections_aligned.jsonl
  -> 输出 vis/<camera>/<pcd_timestamp>.jpg
  -> 输出 vis_by_image/<camera>/<rgb_timestamp>.jpg
  -> 每个 projected_boxes[] 保存 8 个 corners_2d

第二阶段：小样本与全量验证
  -> 小样本验证通过
  -> 全量结果已输出到 vis_projection_newstruct_final/
```

下一步进入：

```text
第三阶段 A：实现同图叠加可视化脚本
```

第三阶段 A 只负责生成 `vis/<camera>/<pcd_timestamp>.jpg` 叠加图，暂不计算 IoU，也不输出 TP/FP/FN。

## 当前程序关系

当前投影相关程序位于：

```text
/home/c64508/桌面/final_3dproject/project_task/build_projection_index.py
/home/c64508/桌面/final_3dproject/project_task/project_boxes_to_images_newstruct.py
/home/c64508/桌面/final_3dproject/project_task/project_boxes_to_images_final.py
```

它们分工如下：

- `build_projection_index.py`：生成时间对齐索引 `aligned_index.json`。
- `project_boxes_to_images_newstruct.py`：读取 `aligned_index.json`，把 `pcd/prelabel-model/*.json` 中的 3D 检测框投影到 `jpg/<camera>/<rgb_timestamp>.jpg`，输出投影可视化图和 `projection_summary.json`。
- `project_boxes_to_images_final.py`：在保留投影可视化图的基础上，额外输出 `projections_aligned.jsonl`，并按 `vis/` 和 `vis_by_image/` 保存两套投影可视化图。

当前链路为：

```text
pcd/meta.json
struct_json/meta.json
jpg/meta.json
  -> build_projection_index.py
  -> aligned_index.json
  -> project_boxes_to_images_final.py
  -> vis_projection_newstruct_final/vis/<camera>/<pcd_timestamp>.jpg
  -> vis_projection_newstruct_final/vis_by_image/<camera>/<rgb_timestamp>.jpg
  -> vis_projection_newstruct_final/projections_aligned.jsonl
  -> vis_projection_newstruct_final/projection_summary.json
```

## 已有输入输出

以当前数据集为例，关键输入包括：

```text
aligned_index.json
pcd/prelabel-model/<pcd_timestamp>.json
struct_json/calib/<calib_timestamp>.json
jpg/<camera>/<rgb_timestamp>.jpg
aligned_grounding_dino_b/detections_aligned.jsonl
```

当前投影程序已经输出：

```text
vis_projection_newstruct_final/
  aligned_index.json
  projection_summary.json
  projections_aligned.jsonl
  vis/
    camera_front_wide/
      <pcd_timestamp>.jpg
    camera_rear/
      <pcd_timestamp>.jpg
  vis_by_image/
    camera_front_wide/
      <rgb_timestamp>.jpg
    camera_rear/
      <rgb_timestamp>.jpg
  ...
```

其中 `projection_summary.json` 目前主要保存每个 `pcd_timestamp + camera` 的统计信息：

```json
{
  "pcd_timestamp": 1782286937435038208,
  "boxes_total": 1,
  "cameras": {
    "camera_front_wide": {
      "input_image": "2069758074335653889/jpg/camera_front_wide/1782286937627422849.jpg",
      "output_image": "2069758074335653889/vis_projection_newstruct_final/vis/camera_front_wide/1782286937435038208.jpg",
      "boxes_total": 1,
      "boxes_projected": 1,
      "boxes_skipped_behind_camera": 0,
      "boxes_skipped_outside_image": 0,
      "image_timestamp": 1782286937627422849,
      "image_delta_ms": 192.384641
    }
  }
}
```

## 已解决的投影结构缺口

旧版 `projection_summary.json` 能说明某个相机视角投影了几个框，也能定位投影可视化图片，但它没有保存每个 3D 框投影后的具体 2D 框坐标。

因此旧流程只能做：

```text
GroundingDINO 检测图
投影可视化图
人工并排查看
```

还不能可靠做：

```text
检测框 bbox_xyxy vs 投影框 bbox_xyxy
IoU 计算
按类别匹配
漏检/误检统计
同图叠加两类框
```

现在这一缺口已经由 `project_boxes_to_images_final.py` 解决。新流程会输出 `projections_aligned.jsonl`，其中每个成功投影框都包含 `bbox_xyxy` 和 8 个 `corners_2d`。

## 投影输出规范

`build_projection_index.py` 暂时不需要大改。它已经负责生成统一的 `aligned_index.json`，这个文件应继续作为全部流程的主对齐索引。

`project_boxes_to_images_final.py` 已经实现以下规则：

1. 保留投影可视化图输出，并将可视化目录拆成 `vis/` 和 `vis_by_image/`。
2. 在投影每个 3D box 时，记录该 box 投影后的 2D 信息。
3. 为每个 `pcd_timestamp + camera` 输出一条结构化 JSONL 记录。
4. 输出字段与 `detections_aligned.jsonl` 尽量保持同构，方便后续 join。

主输出：

```text
vis_projection_newstruct_final/projections_aligned.jsonl
```

每一行对应一个：

```text
pcd_timestamp + camera
```

## 推荐输出目录结构

建议继续使用数据集根目录下的 `vis_projection_newstruct_final` 作为新版 3D 投影结果目录：

```text
/home/c64508/桌面/dataset/2069758074335653889/vis_projection_newstruct_final/
  aligned_index.json
  projection_summary.json
  projections_aligned.jsonl
  vis/
    camera_front_wide/
      1782286937435038208.jpg
    camera_rear/
      1782286937435038208.jpg
  vis_by_image/
    camera_front_wide/
      1782286937627422849.jpg
    camera_rear/
      1782286937507375960.jpg
  ...
```

其中：

- `vis/<camera>/<pcd_timestamp>.jpg`：按 PCD 主时间戳保存，是后续与 `detections_aligned.jsonl`、对比可视化和投影统计对齐的主可视化目录。
- `vis_by_image/<camera>/<rgb_timestamp>.jpg`：按原始 RGB 图像时间戳保存，方便从图像本身出发检查投影效果，也和模型检测输出中的 `vis_by_image/` 约定保持一致。
- 如果同一张 RGB 图像被多个 PCD frame 复用，`vis_by_image/` 可能发生同名覆盖。因此第一阶段更推荐把它作为快速检查目录；严谨对齐仍以 `vis/` 和 `projections_aligned.jsonl` 为准。

后续模型对比脚本可以输出到独立目录，例如：

```text
/home/c64508/桌面/dataset/2069758074335653889/compare_grounding_dino_b_projection/
  manifest.json
  overlay_summary.json
  vis/
    camera_front_wide/
      1782286937435038208.jpg
```

如果后续加入 IoU 和指标计算，可以在同一个目录继续增加：

```text
matches_aligned.jsonl
metrics_summary.json
```

## projections_aligned.jsonl 推荐格式

`projections_aligned.jsonl` 建议与模型检测输出保持相似结构：

```json
{
  "dataset_id": "2069758074335653889",
  "pcd_timestamp": 1782286937435038208,
  "camera": "camera_front_wide",
  "rgb_timestamp": 1782286937627422849,
  "image_delta_ms": 192.384641,
  "valid_time_match": false,
  "alignment_reason": "delta_ms_exceeds_threshold",
  "rgb_image": "/home/c64508/桌面/dataset/2069758074335653889/jpg/camera_front_wide/1782286937627422849.jpg",
  "projection_image": "/home/c64508/桌面/dataset/2069758074335653889/vis_projection_newstruct_final/vis/camera_front_wide/1782286937435038208.jpg",
  "image_width": 3840,
  "image_height": 2160,
  "source": {
    "name": "prelabel-model",
    "family": "projected_3d_detection",
    "det_json": "/home/c64508/桌面/dataset/2069758074335653889/pcd/prelabel-model/1782286937435038208.json",
    "box_source_frame": "lidar_top_GT",
    "transform_chain": "box_source_to_ego_to_camera_to_pixel"
  },
  "params": {
    "score_threshold": 0.5,
    "min_depth": 0.1
  },
  "projection": {
    "boxes_total": 1,
    "boxes_projected": 1,
    "boxes_skipped_behind_camera": 0,
    "boxes_skipped_outside_image": 0,
    "skipped": false,
    "reason": null
  },
  "projected_boxes": [
    {
      "projection_id": "projection:1782286937435038208:camera_front_wide:0",
      "object_id": "xxx",
      "class_name": "Car",
      "class_id": 0,
      "label_raw": "car",
      "score": 0.91,
      "bbox_xyxy": [1910.2, 996.8, 1995.4, 1076.1],
      "bbox_xyxy_unclipped": [1910.2, 996.8, 1995.4, 1076.1],
      "corners_2d": [
        [1995.4, 1002.1],
        [1988.7, 1076.1],
        [1910.2, 1068.5],
        [1917.6, 996.8],
        [1991.8, 1008.9],
        [1984.2, 1071.4],
        [1914.5, 1065.2],
        [1920.9, 1001.7]
      ],
      "depth_range": {
        "min": 8.2,
        "max": 10.4
      },
      "bbox_3d": {
        "center": [0.0, 0.0, 0.0],
        "size_lwh": [4.5, 1.8, 1.6],
        "heading": 1.57
      },
      "visibility": {
        "projected": true,
        "skip_reason": null
      },
      "extra": {}
    }
  ]
}
```

说明：

- `bbox_xyxy`：裁剪到图像范围内的 2D 外接框，后续 IoU 推荐使用它。
- `bbox_xyxy_unclipped`：原始投影外接框，可能超出图像边界，便于排查。
- `corners_2d`：必须保存 8 个 3D box 角点投影到 2D 像素坐标系后的坐标，用于完整复原 3D 投影线框。
- `corners_2d` 的顺序必须与投影程序中 `box_to_corners(...)` 生成角点的顺序一致，也就是与 `BOX_EDGES` 使用的角点编号一致。这样后续可直接用同一组边连接规则重画线框。
- `projected_boxes`：只放成功投影且与图像有交集的框。
- 如果希望调试被跳过的框，可额外加 `skipped_boxes`，保存 `behind_camera` 或 `outside_image` 的原因。

## 类别统一

为了能和 GroundingDINO 输出对比，投影结果中的类别也应归一到同一张类别表：

```text
0 Car
1 Pedestrian
2 Cyclist
3 Van
4 Traffic_cone
```

投影程序当前从 3D 检测 JSON 读取：

```python
category = str(properties.get("type", "unknown")).lower()
```

建议增加一个类别归一化函数：

```text
car -> Car
vehicle -> Car
pedestrian/person -> Pedestrian
cyclist/bicyclist -> Cyclist
van/minivan -> Van
traffic_cone/traffic cone/cone -> Traffic_cone
```

无法归一到五类的框可以：

1. 默认跳过，不进入 `projected_boxes`。
2. 或保留为 `class_name: "Unknown"`，但后续指标计算时不参与五类对比。

推荐第一阶段先跳过五类之外的对象，保证对比结果语义干净。

## 投影程序内部改动点

当前核心函数为：

```text
render_camera_image(...)
project_index(...)
```

建议改动：

1. 把 `render_camera_image(...)` 从“只画图并返回统计”改为“画图 + 返回统计 + 返回每个 projected_box”。
2. 在 box 循环中，成功通过深度和画幅过滤后，计算：

```text
corners_2d，必须为 8 个 [x, y] 点
bbox_xyxy_unclipped = min/max(corners_2d)
bbox_xyxy = clip 到 [0,width-1] 和 [0,height-1]
depth_range
class_name/class_id
object_id/score/bbox_3d
```

3. 在 `project_index(...)` 中，为每个 camera 写入一条 `projections_aligned.jsonl`。
4. 可视化图保存两份：`vis/<camera>/<pcd_timestamp>.jpg` 和 `vis_by_image/<camera>/<rgb_timestamp>.jpg`。两者图像内容可以相同，但命名索引不同。
5. `projection_summary.json` 继续保留统计用途，不建议把每个 box 的详细信息塞进 summary，否则文件会过大且不方便流式读取。

## 第三阶段拆分方案

投影程序输出 `projections_aligned.jsonl` 后，第三阶段建议拆成两个小阶段完成。

这样做的原因是：同图叠加可视化只依赖对齐键和框坐标，逻辑确定；而 IoU 匹配策略还需要进一步讨论，例如是否按类别匹配、是否使用投影 2D 外接框、如何处理一对多和多对一匹配、使用贪心还是匈牙利匹配等。

### 第三阶段 A：同图叠加可视化

先实现一个只负责叠加可视化的脚本，例如：

```text
scripts/visualize_projection_vs_model.py
```

输入：

```bash
python scripts/visualize_projection_vs_model.py \
  --dataset-root /home/c64508/桌面/dataset/2069758074335653889 \
  --projection-jsonl /home/c64508/桌面/dataset/2069758074335653889/vis_projection_newstruct_final/projections_aligned.jsonl \
  --model-jsonl /home/c64508/桌面/dataset/2069758074335653889/aligned_grounding_dino_b/detections_aligned.jsonl \
  --output-dir /home/c64508/桌面/dataset/2069758074335653889/compare_grounding_dino_b_projection \
  --save-vis
```

对齐键：

```text
pcd_timestamp + camera
```

可选再校验：

```text
rgb_timestamp
rgb_image
image_width
image_height
```

输出：

```text
compare_grounding_dino_b_projection/
  manifest.json
  overlay_summary.json
  vis/<camera>/<pcd_timestamp>.jpg
```

此阶段不计算 IoU，不输出 TP/FP/FN，不输出 `matches_aligned.jsonl`。它只解决一个问题：

```text
同一张原始 RGB 图上，3D 投影 reference 框和模型检测框是否能正确叠加显示
```

### 第三阶段 B：IoU 匹配和指标

在确认叠加可视化没有明显对齐问题后，再实现或扩展指标脚本，例如：

```text
scripts/compare_aligned_detections.py
```

输入：

```bash
python scripts/compare_aligned_detections.py \
  --dataset-root /home/c64508/桌面/dataset/2069758074335653889 \
  --projection-jsonl /home/c64508/桌面/dataset/2069758074335653889/vis_projection_newstruct_final/projections_aligned.jsonl \
  --model-jsonl /home/c64508/桌面/dataset/2069758074335653889/aligned_grounding_dino_b/detections_aligned.jsonl \
  --output-dir /home/c64508/桌面/dataset/2069758074335653889/compare_grounding_dino_b_projection \
  --iou-threshold 0.5
```

对齐键：

```text
pcd_timestamp + camera
```

可选再校验：

```text
rgb_timestamp
rgb_image
image_width
image_height
```

输出：

```text
compare_grounding_dino_b_projection/
  manifest.json
  matches_aligned.jsonl
  metrics_summary.json
```

此阶段再负责：

```text
IoU 矩阵
匹配策略
TP/FP/FN
precision / recall / f1
mean IoU
```

## 可视化规则

推荐同图叠加，而不是只并排看两张图：

```text
底图：原始 RGB 图像
3D 投影框：使用实线或较粗线
GroundingDINO 框：使用虚线或较细线
第三阶段 A 暂不显示匹配状态，也不显示 IoU。
```

颜色建议沿用五类固定颜色，保证不同模型结果一致：

```text
Car: 蓝色
Pedestrian: 绿色
Cyclist: 青色
Van: 黄色
Traffic_cone: 红色
```

为了避免画面过乱，第三阶段 A 先只画：

```text
3D 投影 reference bbox：实线 + label + score
模型检测 bbox：虚线或较细线 + label + score
```

第三阶段 B 确定 IoU 策略后，再考虑画：

```text
3D corners_2d 线框，要求每个投影框必须有 8 个 corners_2d 点
匹配成功：显示 IoU
未匹配投影框：标记为 missed/reference_only
未匹配模型框：标记为 extra/model_only
匹配连线
IoU 数字
```

## 数值对比规则

数值对比规则留到第三阶段 B 再最终确定。初步建议仍然从 2D bbox 级别对比开始：

1. 按 `pcd_timestamp + camera` 取同一张图。
2. 按 `class_id` 分组。
3. 对每一类计算投影框与模型框之间的 IoU 矩阵。
4. 使用贪心匹配或匈牙利匹配。
5. IoU >= 阈值的配对记为 TP。
6. 未匹配投影框记为 FN。
7. 未匹配模型框记为 FP。

输出指标：

```text
per_camera precision / recall / f1
per_class precision / recall / f1
overall precision / recall / f1
mean_iou_of_matches
```

注意：这里的 “GT” 实际是 3D 检测结果投影，不一定是真值标注。因此文档和字段中建议称为：

```text
projection reference
projected_3d_detection
```

不要直接称为 ground truth，除非确认 `pcd/prelabel-model` 就是人工真值。

## 推荐实施顺序

第一阶段：修改投影输出结构，已完成

```text
project_boxes_to_images_final.py
  -> 保留原可视化图
  -> 新增 projections_aligned.jsonl
  -> 每条记录包含 projected_boxes
```

第二阶段：小样本与全量验证，已完成

```text
max_frames=1 或 max_frames=5
检查 projections_aligned.jsonl
检查投影框 bbox_xyxy 和 8 个 corners_2d 连线是否与现有 jpg 可视化一致
检查 pcd_timestamp + camera 是否能和 detections_aligned.jsonl join
```

第三阶段 A：实现同图叠加可视化脚本，下一步执行

```text
visualize_projection_vs_model.py
  -> 读取 projections_aligned.jsonl
  -> 读取 aligned_grounding_dino_b/detections_aligned.jsonl
  -> 按 pcd_timestamp + camera 对齐
  -> 在原始 RGB 图像上叠加 reference 投影框和模型检测框
  -> 输出 overlay_summary.json 和 vis/<camera>/<pcd_timestamp>.jpg
  -> 暂不计算 IoU/matches/metrics
```

第三阶段 B：实现 IoU 匹配和指标脚本

```text
compare_aligned_detections.py
  -> 读取 projections_aligned.jsonl
  -> 读取 aligned_grounding_dino_b/detections_aligned.jsonl
  -> 计算 IoU/matches
  -> 输出 matches_aligned.jsonl 和 metrics_summary.json
```

第四阶段：扩展到其他模型

```text
YOLO detections_aligned.jsonl
SAM masks_aligned.jsonl
其他 open-vocab detector
```

只要其他模型遵守 `pcd_timestamp + camera` 的对齐协议，就可以复用同一个对比脚本。

## 结论

投影程序结构化输出已经完成，下一步不需要继续修改投影主流程，而是先实现同图叠加可视化脚本。

第三阶段 A 的最小必要目标是：

```text
visualize_projection_vs_model.py
  读取 vis_projection_newstruct_final/projections_aligned.jsonl
  读取 aligned_grounding_dino_b/detections_aligned.jsonl
  按 pcd_timestamp + camera 对齐
  在原始 RGB 图像上画 3D 投影 reference bbox 和 GroundingDINO bbox
  输出 compare_grounding_dino_b_projection/vis/<camera>/<pcd_timestamp>.jpg
  输出 overlay_summary.json
```

完成第三阶段 A 后，再讨论第三阶段 B 的 IoU 策略，用 GroundingDINO 的：

```text
aligned_grounding_dino_b/detections_aligned.jsonl
```

和 3D 投影的：

```text
vis_projection_newstruct_final/projections_aligned.jsonl
```

做稳定的一对一匹配、IoU 统计和指标汇总。
