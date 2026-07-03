# 多模型检测结果与投影结果对齐方案

## 目标

以数据集目录为单位，例如：

```text
/home/c64508/桌面/dataset/2069758074335653889
```

复用已有的 PCD 主时间戳对齐结果，让 GroundingDINO、YOLO、SAM 或其他检测模型都在同一批 RGB 图像上推理，并将结果按统一格式保存，便于后期与 `vis_projection_newstruct` 下的 3D 投影结果做可视化和数值对比。

核心目标不是只服务某一个模型，而是建立一套通用协议：

```text
PCD timestamp + camera
  -> 对齐到 RGB input_image
  -> 目标检测模型推理
  -> 输出统一 JSON
  -> 可选输出可视化图
  -> 后期与 3D 投影结果/其他模型结果对比
```

## 已有数据

以 `2069758074335653889` 为例，关键输入文件包括：

```text
aligned_index.json
vis_projection_newstruct/aligned_index.json
vis_projection_newstruct/projection_summary.json
jpg/<camera>/<rgb_timestamp>.jpg
vis_projection_newstruct/<camera>/<pcd_timestamp>.jpg
```

其中：

- `aligned_index.json`：记录 PCD 主时间戳与邻近 RGB 图像、标定文件、检测文件之间的时间对齐关系。它是目标检测脚本应优先读取的主索引。
- `vis_projection_newstruct/aligned_index.json`：与根目录下的对齐索引内容一致或等价，可作为投影流程使用的对齐索引。
- `vis_projection_newstruct/projection_summary.json`：投影流程的派生结果，记录每个 PCD timestamp 在每个 camera 下使用了哪张 RGB 图像，以及投影可视化图保存在哪里。它应作为可选增强输入，而不是检测流程的必要输入。
- `jpg/<camera>/<rgb_timestamp>.jpg`：原始 RGB 图像，是目标检测模型应该推理的图像。
- `vis_projection_newstruct/<camera>/<pcd_timestamp>.jpg`：已经画了 3D 投影框的可视化图，不应该作为检测模型输入，否则会污染检测结果。

## 关键对齐关系

`aligned_index.json` 中每个 frame 以 PCD timestamp 为主索引：

```json
{
  "pcd_timestamp": 1782286937435038208,
  "images": {
    "camera_front_wide": {
      "timestamp": 1782286937627422849,
      "path": "jpg/camera_front_wide/1782286937627422849.jpg",
      "delta_ms": 192.384641,
      "calib_camera_name": "camera_front_wide",
      "valid_time_match": false,
      "reason": null
    }
  }
}
```

因此对齐链路为：

```text
pcd_timestamp
  + camera
  -> aligned_index.frames[*].images[camera]
  -> path
  -> 检测模型推理 path 对应的原始 RGB 图像
  -> 将检测结果写回 pcd_timestamp + camera 记录
```

如果同时提供 `projection_summary.json`，则可以额外补充投影可视化图路径：

```text
projection_summary.frames[*].cameras[camera].output_image
```

注意：原始 RGB 图像文件名是 RGB timestamp，而投影可视化图文件名通常是 PCD timestamp。后续输出 JSON 时必须同时保留这两个时间戳。

## 总体方案

检测脚本不直接遍历 `jpg/` 下所有图片，而是优先读取：

```text
aligned_index.json
```

这样即使数据只完成了时间对齐、尚未进行 3D 投影，目标检测模型也知道应该处理哪些 PCD timestamp 对应的哪些相机图像。

`projection_summary.json` 作为可选输入使用：

```text
vis_projection_newstruct/projection_summary.json
```

当它存在时，可以把投影可视化图路径、投影统计信息补充进检测结果，方便后期对比。

推荐流程：

1. 读取 `aligned_index.json`。
2. 遍历每个 `frame`。
3. 对每个 frame 下的 6 路 camera 读取 `images[camera].path`。
4. 建立唯一 RGB 图像列表，避免同一张 RGB 被多个 PCD frame 重复检测。
5. 对唯一 RGB 图像执行模型推理。
6. 将检测结果按 `pcd_timestamp + camera` 展开写入对齐结果 JSON。
7. 如果提供了 `projection_summary.json`，补充 `projection_image`、投影统计等字段。
8. 可选：生成可视化图。
9. 保存 summary，记录模型名、阈值、输入索引、输出数量等元信息。

## 为什么要去重检测

在当前数据集中：

```text
500 个 PCD frame × 6 路 camera = 3000 个 PCD-camera 对齐项
```

但对应的唯一 RGB 图像数量可能略少，因为少量 RGB 图像会被多个邻近 PCD timestamp 复用。

因此最佳实践是：

```text
唯一 RGB 图像只推理一次
同一检测结果可挂接到多个 pcd_timestamp + camera 对齐项
```

这样可以减少重复计算，也能保持严格对齐。

## 输入设计

通用检测脚本建议命名为：

```text
2d_detect/run_aligned_model_detection.py
```

建议支持参数：

```bash
python 2d_detect/run_aligned_model_detection.py \
  --dataset-root /home/c64508/桌面/dataset/2069758074335653889 \
  --aligned-index /home/c64508/桌面/dataset/2069758074335653889/aligned_index.json \
  --projection-summary /home/c64508/桌面/dataset/2069758074335653889/vis_projection_newstruct/projection_summary.json \
  --model-name grounding_dino_b \
  --save-vis
```

其中：

- `--aligned-index` 是必需参数。
- `--projection-summary` 是可选参数，仅在需要关联投影可视化图或投影统计时使用。
- `--output-dir` 是可选参数。默认输出到 `<dataset_root>/aligned_<model_name>/`，例如 `/home/c64508/桌面/dataset/2069758074335653889/aligned_grounding_dino_b/`。临时测试时也可以手动指定为 `/home/c64508/桌面/dataset/2069758074335653889/aligned_grounding_dino_b_test/`。

不同模型可扩展自己的参数，例如 GroundingDINO：

```bash
--grounding-config /home/c64508/桌面/compare_model_detection/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py
--grounding-checkpoint /home/c64508/桌面/compare_model_detection/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth
--prompt "car . pedestrian . cyclist . van . traffic cone ."
--box-threshold 0.25
--text-threshold 0.25
```

YOLO 可扩展：

```bash
--yolo-weights /path/to/yolo.pt
--conf-threshold 0.25
--iou-threshold 0.7
```

SAM 类模型如果用于分割，可扩展：

```bash
--sam-checkpoint /path/to/sam.pth
--sam-model-type vit_h
--use-box-prompts
```

## 输出目录结构

建议每个模型一次实验输出到数据集根目录下的独立目录。这样检测结果、原始图像、时间对齐文件和投影结果都随同一个数据集归档，后期查看和处理更直接。

```text
/home/c64508/桌面/dataset/2069758074335653889/aligned_grounding_dino_b/
  manifest.json
  detections_aligned.jsonl
  detections_by_image.jsonl
  summary.json
  vis/
    camera_front_wide/
      1782286937435038208.jpg
    camera_rear/
      1782286937435038208.jpg
  logs/
    run.log
```

说明：

- `manifest.json`：本次运行配置，记录模型、权重、阈值、输入索引路径、类别表等。
- `detections_by_image.jsonl`：以唯一 RGB 图像为单位保存原始检测结果。
- `detections_aligned.jsonl`：以 `pcd_timestamp + camera` 为单位保存展开后的对齐结果，是后续模型对比的主文件。
- `summary.json`：保存统计信息。
- `vis/`：可选可视化图，建议使用 PCD timestamp 命名，便于与 `vis_projection_newstruct` 对应。
- `logs/`：可选运行日志。

## JSONL 主输出格式

后期模型对比建议主要使用：

```text
detections_aligned.jsonl
```

每一行是一条 `pcd_timestamp + camera` 记录。

推荐结构：

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
  "projection_image": "/home/c64508/桌面/dataset/2069758074335653889/vis_projection_newstruct/camera_front_wide/1782286937435038208.jpg",
  "image_width": 3840,
  "image_height": 2160,
  "model": {
    "name": "grounding_dino_b",
    "family": "open_vocab_detection",
    "checkpoint": "groundingdino_swinb_cogcoor.pth",
    "config": "GroundingDINO_SwinB_cfg.py"
  },
  "params": {
    "prompt": "car . pedestrian . cyclist . van . traffic cone .",
    "box_threshold": 0.25,
    "text_threshold": 0.25
  },
  "detections": [
    {
      "det_id": "grounding_dino_b:1782286937435038208:camera_front_wide:0",
      "class_name": "Car",
      "class_id": 0,
      "label_raw": "car",
      "score": 0.692897,
      "bbox_xyxy": [3059.488525, 999.482422, 3209.740479, 1067.85498],
      "bbox_cxcywh_norm": [0.816306, 0.47855, 0.039128, 0.031654],
      "segmentation": null,
      "extra": {}
    }
  ]
}
```

字段说明：

- `pcd_timestamp`：主时间戳，来自 PCD。
- `camera`：相机名。
- `rgb_timestamp`：实际被检测的 RGB 图像时间戳。
- `image_delta_ms`：RGB 与 PCD 的时间差，来自 `aligned_index.json` 中对应相机图像的 `delta_ms` 字段。
- `valid_time_match`：该相机图像是否满足对齐阈值。
- `alignment_reason`：如果对齐无效，记录原因。
- `rgb_image`：检测模型实际输入图像。
- `projection_image`：3D 投影可视化图，作为后续叠加对比参考。仅当提供 `projection_summary.json` 且存在对应投影图时填写，否则为 `null`。
- `detections`：该模型在该 RGB 图上的检测结果。

## 类别规范

统一类别表建议固定为：

```json
[
  {"class_id": 0, "class_name": "Car"},
  {"class_id": 1, "class_name": "Pedestrian"},
  {"class_id": 2, "class_name": "Cyclist"},
  {"class_id": 3, "class_name": "Van"},
  {"class_id": 4, "class_name": "Traffic_cone"}
]
```

GroundingDINO 是开放词表模型，原始输出可能是：

```text
car
car van
pedestrian
person
traffic cone
```

因此需要保留两个字段：

```json
{
  "class_name": "Van",
  "label_raw": "car van"
}
```

其中：

- `label_raw`：模型原始输出。
- `class_name`：映射到统一类别表后的结果。

推荐类别映射规则：

```text
traffic cone / cone -> Traffic_cone
cyclist / bicyclist / person riding bicycle -> Cyclist
van / minivan -> Van
pedestrian / person -> Pedestrian
car / vehicle -> Car
```

当一个原始短语包含多个类别词时，应优先匹配更具体类别：

```text
Traffic_cone > Cyclist > Van > Pedestrian > Car
```

## 坐标规范

所有检测模型输出必须统一到两种坐标：

```json
{
  "bbox_xyxy": [x1, y1, x2, y2],
  "bbox_cxcywh_norm": [cx, cy, w, h]
}
```

其中：

- `bbox_xyxy`：像素坐标，左上角和右下角，适合画框和计算 IoU。
- `bbox_cxcywh_norm`：归一化中心点宽高，适合跨分辨率比较和部分训练格式转换。

约定：

```text
x1, x2 范围：[0, image_width]
y1, y2 范围：[0, image_height]
cx, cy, w, h 范围：[0, 1]
```

后续所有模型都必须输出同一坐标格式。

## 分割模型扩展

SAM 或其他分割模型可以沿用同一结构，只需要给 detection 增加 `segmentation` 字段：

```json
{
  "segmentation": {
    "type": "rle",
    "counts": "...",
    "size": [2160, 3840]
  }
}
```

如果暂时只比较检测框，可先保存由 mask 外接矩形得到的 `bbox_xyxy`，同时保留 mask 信息：

```json
{
  "bbox_xyxy": [x1, y1, x2, y2],
  "segmentation": {
    "type": "polygon",
    "points": [[x, y], [x, y]]
  }
}
```

## 可视化输出规范

检测模型可视化图建议保存到：

```text
outputs/<run_name>/vis/<camera>/<pcd_timestamp>.jpg
```

命名使用 `pcd_timestamp`，而不是 `rgb_timestamp`，原因是：

- 便于和 `vis_projection_newstruct/<camera>/<pcd_timestamp>.jpg` 一一对应。
- 便于按 PCD 时间轴浏览所有模型结果。
- 即使同一 RGB 被多个 PCD 复用，也能生成每个 PCD 对齐视角对应的可视化。

可视化图建议支持三种模式：

```text
model_only      只画模型检测框
projection_only 只使用已有投影图
overlay         在原始 RGB 上同时画投影框和模型框
```

当前阶段可先实现 `model_only`。后续如果投影流程能输出具体 2D 投影框坐标，再实现 `overlay` 和 IoU 对比。

## 与 3D 投影结果的关系

当前 `projection_summary.json` 主要保存了：

```text
input_image
output_image
boxes_total
boxes_projected
boxes_skipped_behind_camera
boxes_skipped_outside_image
image_timestamp
image_delta_ms
```

它目前不一定保存每个投影框的具体 2D 坐标。

因此：

1. 如果只做可视化并排对比，可以直接使用：

```text
vis_projection_newstruct/<camera>/<pcd_timestamp>.jpg
outputs/<run_name>/vis/<camera>/<pcd_timestamp>.jpg
```

2. 如果要做 IoU 或召回率等数值对比，需要投影流程额外保存：

```json
{
  "projected_boxes": [
    {
      "source_3d_box_id": "...",
      "class_name": "...",
      "bbox_xyxy": [x1, y1, x2, y2],
      "corners_2d": [[x, y], [x, y]],
      "valid": true
    }
  ]
}
```

后续建议新增一个投影框结构化输出文件，例如：

```text
vis_projection_newstruct/projected_boxes_aligned.jsonl
```

其主键也应为：

```text
pcd_timestamp + camera
```

## 通用模型适配器设计

代码层面建议拆成两层：

```text
对齐调度层
模型适配层
```

对齐调度层负责：

- 读取 `aligned_index.json`
- 可选读取 `projection_summary.json`
- 生成唯一 RGB 图像列表
- 调用模型适配器推理
- 将结果展开到 `pcd_timestamp + camera`
- 保存 JSONL 和可视化

模型适配层负责：

- 加载模型
- 输入一张图像
- 输出统一 detection list

统一接口建议：

```python
class DetectorAdapter:
    model_name: str

    def predict(self, image_path: str) -> list[dict]:
        ...
```

每个 detection 必须返回：

```python
{
    "class_name": "Car",
    "class_id": 0,
    "label_raw": "car",
    "score": 0.9,
    "bbox_xyxy": [x1, y1, x2, y2],
    "bbox_cxcywh_norm": [cx, cy, w, h],
    "segmentation": None,
    "extra": {}
}
```

GroundingDINO、YOLO、SAM 都通过 adapter 转成这个统一格式。

## 对齐筛选策略

脚本应支持是否只处理有效时间匹配样本：

```bash
--valid-time-only
```

默认建议不丢弃样本，而是在输出 JSON 里保留：

```json
{
  "valid_time_match": false,
  "alignment_reason": "delta_ms_exceeds_threshold",
  "image_delta_ms": 192.384641
}
```

这样后期分析时可以自由筛选：

```text
只看 valid_time_match = true
只看 image_delta_ms < 20ms
按不同时间差分桶统计
```

## 推荐实现顺序

第一阶段：GroundingDINO 对齐输出

- 读取 `aligned_index.json`
- 根据 `frames[*].images[camera].path` 去重 RGB 图像
- GroundingDINO 推理
- 输出 `detections_by_image.jsonl`
- 输出 `detections_aligned.jsonl`
- 可选输出 `vis/<camera>/<pcd_timestamp>.jpg`
- 可选读取 `projection_summary.json`，补充投影可视化图路径

第二阶段：YOLO 适配器

- 增加 YOLO adapter
- 复用同一个对齐调度层
- 输出同样格式 JSON

第三阶段：SAM 或分割模型

- 增加 segmentation 字段
- 可选使用 GroundingDINO/YOLO 检测框作为 SAM prompt
- 输出 bbox + mask

第四阶段：对比工具

- 读取多个模型的 `detections_aligned.jsonl`
- 按 `pcd_timestamp + camera` 对齐
- 生成并排图或叠加图
- 如果有投影框坐标，计算 IoU/召回/漏检/误检

## 最终结论

需要编写的 Python 脚本本质上是一个通用的“对齐检测导出器”：

```text
告诉目标检测模型：
  1. 根据 aligned_index 应该检测哪些 RGB 图
  2. 检测结果如何按 pcd_timestamp + camera 保存
  3. JSON 结果如何统一字段
  4. 可视化图如何命名和落盘
  5. 如果存在 projection_summary，如何关联投影可视化与投影统计
```

只要这套协议固定下来，后续 GroundingDINO、YOLO、SAM 或其他模型都可以通过 adapter 接入，并在同一套对齐索引下做公平比较。
