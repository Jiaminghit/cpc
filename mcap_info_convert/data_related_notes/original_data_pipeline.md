# 源程序架构
这个程序本质上是一个“从 S3 兼容对象存储下载点云数据 → 执行 3D 目标检测 → 将结果上传回对象存储”的 Argo 工作流节点。

核心入口在 [prelabel_main.py](/home/c64508/桌面/prelabel_model/prelabel_model/prelabel_main.py:91)。

## 整体流水线

```text
Argo 传入 task_id + param_url
               │
               ▼
解析 bucket 和数据集前缀
               │
               ▼
下载 prefix/meta.json
               │
               ▼
读取 frame_map，逐帧下载传感器文件
               │
               ▼
从中找到 .pcd 点云文件
               │
               ▼
点云预处理 + 3D 检测模型推理
               │
               ▼
生成每帧预标注 JSON 并上传 S3
               │
               ▼
更新 meta.json 中的 preann 字段
               │
               ▼
Kafka 发送 SUCCESS / FAILED
```

## 1. 程序接收什么参数

实际命令行参数是：

```bash
python -m prelabel_model.prelabel_main \
  --task_id <任务ID> \
  --param_url <bucket/path/to/dataset>
```

需要注意：文件顶部注释写的是 `--root_path`，但真正解析的参数是 `--param_url`，见 [prelabel_main.py:210](/home/c64508/桌面/prelabel_model/prelabel_model/prelabel_main.py:210)。两者表达的是同一个东西，但名称没有同步。

例如：

```text
param_url =
st-cq-dev-sacp-label-data/label_dataset/2026/06/17/xxx
```

解析后得到：

```text
bucket = st-cq-dev-sacp-label-data
prefix = label_dataset/2026/06/17/xxx
```

也支持：

```text
s3://st-cq-dev-sacp-label-data/label_dataset/2026/06/17/xxx
```

解析逻辑位于 [utils.py:17](/home/c64508/桌面/prelabel_model/prelabel_model/utils.py:17)。

## 2. meta.json 是怎么得到的

`meta.json` 不是这个节点创建的，而是默认已经由上游数据准备节点放进对象存储。

程序固定下载：

```text
s3://<bucket>/<prefix>/meta.json
```

例如：

```text
s3://st-cq-dev-sacp-label-data/
  label_dataset/2026/06/17/xxx/meta.json
```

对应代码：

```python
manager = MetaManager(s3, bucket, prefix, ...)
meta = manager.load()
```

`MetaManager.load()` 内部执行：

```python
download_file(bucket, f"{prefix}/meta.json", local_file)
```

见 [meta_manager.py:67](/home/c64508/桌面/prelabel_model/common/meta_manager.py:67)。

如果这个路径下不存在 `meta.json`，程序会直接失败，并由 `guarded_main()` 发送 Kafka `FAILED` 通知。

### meta.json 的作用

它相当于整个数据集的“目录清单”。仓库里没有具体样例，但从代码看，其核心结构大致是：

```json
{
  "dataset_name": "example",
  "sensors": [
    {"name": "p128_0"},
    {"name": "camera_front"}
  ],
  "frame_count": 2,
  "frame_map": [
    {
      "timestamp": 1710000000000000000,
      "files": {
        "p128_0": "data/1710000000000000000.pcd",
        "camera_front": "data/1710000000000000000.jpg"
      },
      "preann": {}
    }
  ]
}
```

最重要的是：

- `frame_map`：帧列表。
- `timestamp`：该帧时间戳。
- `files`：传感器名称到文件路径的映射。
- `preann`：各预标注模型生成的结果路径。

## 3. 会接收到哪些类型的数据

从代码层面可以分为四类。

### ① meta.json：数据集索引

作用是告诉程序：

- 一共有多少帧；
- 每帧时间戳是什么；
- 每帧包含哪些传感器文件；
- 每个文件在 S3 中的位置。

它只用于调度和记录，不直接送入检测模型。

### ② 帧级传感器文件

程序会遍历每一帧的整个 `files`：

```python
for sensor_name, file_path in files.items():
    s3.download_file(...)
```

也就是说，不论是：

- `.pcd` 点云；
- `.jpg`、`.png` 图像；
- `.json`；
- 其他扩展名；

只要出现在 `frame["files"]` 里，当前入口都会尝试下载。

但是，当前模型真正使用的只有第一个 `.pcd` 文件：

```python
pcd_path = _find_first_by_suffix(sensor_files, ".pcd")
```

见 [detector.py:78](/home/c64508/桌面/prelabel_model/prelabel_model/detector.py:78)。

因此：

| 数据 | 是否下载 | 是否进入当前模型 |
|---|---:|---:|
| PCD 激光雷达点云 | 是 | 是 |
| 相机图片 | 是 | 否 |
| 其他传感器文件 | 是 | 否 |
| 帧级普通 JSON | 是 | 通常否 |

**当前代码不是多模态融合模型，而是点云 3D 检测模型。图片即使被下载，目前也不会参与推理。**

如果一帧中没有成功下载任何 `.pcd` 文件，会抛出：

```text
RuntimeError: no .pcd file found in sensor_files
```

并使整个任务失败，而不是只跳过这一帧。

### ③ 任务级 data_RAW@*.json

程序还会额外扫描：

```text
<prefix>/data/
```

寻找文件名包含：

```text
data_RAW@*.json
```

见 [prelabel_main.py:35](/home/c64508/桌面/prelabel_model/prelabel_model/prelabel_main.py:35)。

如果存在多个候选文件，会排序后取最后一个。这个 JSON 会被添加为：

```python
sensor_files["task_data"] = task_data_json
```

它不是帧级检测数据，主要用于读取激光雷达的虚拟坐标变换，然后对点云进行坐标转换：

```python
points = module.apply_transform(
    points,
    module.load_task_virtual_transform(task_data_json, HESAI_LIDAR_NAME),
)
```

见 [detector.py:119](/home/c64508/桌面/prelabel_model/prelabel_model/detector.py:119)。

如果找不到该文件，程序不会失败，只会跳过虚拟坐标变换。

### ④ 本地模型运行资源

这些不是从本任务 S3 路径下载的，而是通过容器环境变量配置：

- `HESAI_RUNTIME_ROOT`：模型运行时目录；
- `HESAI_DET_CONFIG`：检测配置文件；
- `HESAI_PCD_SCRIPT`：PCD 读取和处理脚本；
- `HESAI_LO_RESULT_TXT`：可选的里程计补偿数据；
- `HESAI_LIDAR_NAME`：默认 `p128_0`。

配置见 [config.py:16](/home/c64508/桌面/prelabel_model/prelabel_model/config.py:16)。

## 4. 每一帧怎么处理

每帧依次执行以下步骤。

### 下载文件

根据 `files` 中的路径生成 S3 key。

相对路径：

```text
data/xxx.pcd
```

会解析为：

```text
<prefix>/data/xxx.pcd
```

以 `/` 或 `unique/` 开头的路径被视为桶内绝对 key。

### 加载点云

外部运行时脚本读取 `.pcd`：

```python
points, timestamps, rings, header = module.load_pcd(pcd_path)
```

得到：

- 点坐标及其他点属性；
- 每个点的时间戳；
- 激光雷达线束编号 `rings`；
- PCD 头信息。

### 点云预处理

根据配置，可能依次执行：

1. LO/里程计运动补偿；
2. `project_like_raw2task_p128` 点云投影；
3. 使用 `data_RAW@*.json` 执行虚拟坐标变换；
4. 模型配置中的输入 pipeline。

### 模型推理

构造单帧 batch 后调用：

```python
data_dict = det_node({"det3d_inputs": batch})
```

得到多个 3D 目标实例。

## 5. 推理结果是什么

每帧会产生一个类似 GeoJSON 的 `FeatureCollection`：

```json
{
  "type": "FeatureCollection",
  "metadata": {
    "name": "odpre",
    "version": "1.0",
    "crs": "Ego"
  },
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Box",
        "coordinates": [
          [10.2, 3.4, 1.1],
          [4.5, 1.8, 1.6],
          [0.0, 0.0, 1.57]
        ]
      },
      "properties": {
        "id": "1",
        "type": "car",
        "score": 0.93,
        "velocity": [1.2, 0.1]
      }
    }
  ]
}
```

其中 Box 坐标分别表示：

```text
[x, y, z]                 中心点
[length, width, height]   尺寸
[roll, pitch, heading]    朝向
```

每帧结果上传到：

```text
s3://<bucket>/<prefix>/prelabel-model/<timestamp>.json
```

## 6. meta.json 如何更新

推理成功后，每帧会增加：

```json
{
  "preann": {
    "prelabel-model": "prelabel-model/1710000000000000000.json"
  }
}
```

顶层还会更新：

```json
{
  "has_preannotation": true,
  "preannotation_models": ["prelabel-model"]
}
```

写回之前，`MetaManager` 会：

1. 将原始 `meta.json` 备份到 `meta_backup/`；
2. 重新下载最新版本；
3. 按时间戳合并 `frame_map`；
4. 合并不同节点写入的 `preann`；
5. 上传覆盖原来的 `meta.json`。

见 [meta_manager.py:82](/home/c64508/桌面/prelabel_model/common/meta_manager.py:82)。

## 7. Kafka 通知

全部帧完成并成功写回 `meta.json` 后，发送：

```json
{
  "taskId": "...",
  "status": "SUCCESS",
  "argoNode": "prelabel-model"
}
```

如果下载、推理或上传过程中出现未处理异常，`guarded_main()` 会捕获异常并发送 `FAILED`，见 [notify_util.py:127](/home/c64508/桌面/prelabel_model/common/notify_util.py:127)。

总结来说：当前节点虽然会下载 `meta.json` 中列出的多种传感器文件，但实际推理是“单 PCD 点云 3D 检测”；`meta.json` 来自上游节点，负责描述数据集；`data_RAW@*.json` 是可选的任务级坐标变换配置。

---
