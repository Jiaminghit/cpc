# 接口修改方案

新增独立的 MCAP→PCD 转换模块，并对现有入口做少量接口增强，不要把转换代码直接塞进 `prelabel_main.py`。

原因是每个 MCAP 有数百 MB 到近 4 GB，总计约 12 GB。它是连续记录文件，一个 MCAP 中包含很多帧和多个雷达 topic；应该只顺序读取一次并批量导出，不能在逐帧推理循环里反复打开。

## 一、当前数据情况

目录中有：

- `camera/`：5 个 camera MCAP
- `lidar/`：6 个 lidar MCAP，总计约 12 GB

雷达 MCAP 内至少包含 5 个 topic：

```text
/lidar/pandar
/lidar/qt_front
/lidar/qt_left
/lidar/qt_rear
/lidar/qt_right
```

消息类型是自定义 ROS2 消息：

```text
sdk_msg/msg/PointCloud2V2
```

它和 `sensor_msgs/PointCloud2` 类似，包含：

```text
header
height
width
fields
is_bigendian
point_step
row_step
data
is_dense
```

因此转换器必须根据消息中的 `fields + offset + datatype + point_step` 解码二进制点数据，不能简单假定每个点只有固定的 `x/y/z`。

当前模型配置默认雷达名称是 `p128_0`，而检测模型看起来是 Hesai P128 模型。因此第一阶段应优先验证：

```text
/lidar/pandar  →  p128_0
```

另外四个 `qt_*` 很可能是周视雷达，当前模型未必支持，暂时不应把五路点云随意合并。

## 二、推荐的数据流水线

```text
lidar/*.mcap
      │
      ▼
MCAP 扫描：识别 topic、字段、消息数和时间范围
      │
      ▼
提取 /lidar/pandar 中每条 PointCloud2V2
      │
      ▼
生成一帧一个 PCD
      │
      ▼
生成 meta.json
      │
      ▼
上传 PCD + meta.json 到 S3
      │
      ▼
调用现有 prelabel_main.py
      │
      ▼
3D 检测结果写回 S3
```

## 三、建议新增的文件

### 1. `prelabel_model/mcap_reader.py`

职责：只负责读取和解码 MCAP。

建议接口：

```python
@dataclass
class PointCloudFrame:
    timestamp_ns: int
    topic: str
    frame_id: str
    fields: list
    points: np.ndarray


def inspect_mcap(path: Path) -> McapInfo:
    """返回 topic、消息类型、消息数量、时间范围和点字段。"""


def iter_pointcloud_frames(
    path: Path,
    topic: str,
) -> Iterator[PointCloudFrame]:
    """顺序读取指定 topic 的点云帧。"""
```

项目已经在 [requirements-dev.txt](/home/c64508/桌面/prelabel_model/requirements-dev.txt:1) 中声明：

```text
mcap==1.4.0
mcap-ros2-support==0.5.7
```

官方 `mcap_ros2.reader.read_ros2_messages()` 可以在不安装完整 ROS2 环境的情况下读取 ROS2 MCAP 消息。[官方说明](https://github.com/foxglove/mcap/tree/main/python/mcap-ros2-support)

不过当前宿主机 Python 没有安装这些包，Docker 镜像构建后才会有。

### 2. `prelabel_model/pcd_writer.py`

职责：把解码后的点云保存成模型兼容的 PCD。

建议接口：

```python
def write_pcd(
    output_path: Path,
    points: np.ndarray,
    fields: list[str],
    binary: bool = True,
) -> None:
    ...
```

至少需要确认并保留：

```text
x
y
z
intensity
timestamp/time
ring/channel
```

这里尤其重要：当前检测器会从 PCD 中读取：

```python
points, timestamps, rings, header = module.load_pcd(pcd_path)
```

而且后续投影可能使用 `timestamps` 和 `rings`，见 [detector.py](/home/c64508/桌面/prelabel_model/prelabel_model/detector.py:98)。

所以只输出 `x y z intensity` 可能不够。应该先扫描 MCAP 的真实字段，再建立映射，例如：

```python
FIELD_ALIASES = {
    "timestamp": ["timestamp", "time", "t", "offset_time"],
    "ring": ["ring", "channel", "laser_id"],
    "intensity": ["intensity", "reflectivity"],
}
```

推荐输出 binary PCD。12 GB MCAP 如果转 ASCII PCD，体积和耗时都会明显增大。

### 3. `prelabel_model/mcap_to_pcd.py`

这是批量转换入口，负责串联 reader 和 writer。

建议命令：

```bash
python -m prelabel_model.mcap_to_pcd \
  --input_dir /home/c64508/桌面/dataset/2067268107790897153/lidar \
  --output_dir /home/c64508/桌面/dataset/2067268107790897153/pcd \
  --topic /lidar/pandar \
  --sensor_name p128_0 \
  --generate_meta
```

建议生成：

```text
2067268107790897153/
├── camera/
├── lidar/
├── pcd/
│   └── p128_0/
│       ├── 1781678820000000000.pcd
│       ├── 1781678820100000000.pcd
│       └── ...
└── meta.json
```

转换器还应具备：

- 按时间戳命名，避免跨 MCAP 文件重名；
- 如果目标文件已存在，可跳过，支持断点续转；
- 检查时间戳重复；
- 记录每个 MCAP 的成功帧数和坏帧；
- 使用临时文件写完后原子重命名；
- 支持 `--max_frames`，方便先转换几帧验证；
- 支持 `--inspect-only`，只查看 topic 和字段，不生成 PCD。

## 四、需要新增 meta.json 生成逻辑

当前本地数据目录中没有 `meta.json`，而现有推理入口必须从 S3 下载它。

转换完成后应生成类似：

```json
{
  "schema_version": "1.0",
  "dataset_name": "2067268107790897153",
  "dataset_type_code": "lidar",
  "frame_count": 1000,
  "has_preannotation": false,
  "preannotation_models": [],
  "sensors": [
    {
      "name": "p128_0",
      "type": "lidar",
      "source_topic": "/lidar/pandar"
    }
  ],
  "frame_map": [
    {
      "timestamp": 1781678820000000000,
      "files": {
        "p128_0": "pcd/p128_0/1781678820000000000.pcd"
      },
      "preann": {}
    }
  ]
}
```

可新增：

```text
prelabel_model/meta_builder.py
```

接口：

```python
def build_meta(
    dataset_name: str,
    sensor_name: str,
    frames: list[ConvertedFrame],
) -> dict:
    ...
```

注意必须使用消息自身的采集时间戳，最好取：

```text
header.stamp.sec * 1_000_000_000 + header.stamp.nanosec
```

不要用 MCAP 文件名中的分钟时间，也不要优先使用消息写入 MCAP 的时间。

## 五、现有项目需要修改什么

### 1. `prelabel_main.py`：不负责转换，只增强输入校验

现有入口可以基本保留。建议增加：

- 校验 `frame_map` 是否为空；
- 校验目标传感器 `p128_0` 是否存在；
- 明确选择 PCD，而不是“找到第一个 `.pcd`”；
- 单帧失败策略可配置；
- 不要下载当前模型不用的 camera 文件。

目前代码会下载 `frame["files"]` 中的全部文件：

```python
for sensor_name, file_path in files.items():
```

可以修改为只下载模型需要的传感器：

```python
required_sensors = {HESAI_LIDAR_NAME}

for sensor_name, file_path in files.items():
    if sensor_name not in required_sensors:
        continue
```

这样以后 meta 中加入 camera，也不会白白下载大文件。

### 2. `detector.py`：按传感器名称选 PCD

当前是：

```python
pcd_path = _find_first_by_suffix(sensor_files, ".pcd")
```

见 [detector.py](/home/c64508/桌面/prelabel_model/prelabel_model/detector.py:85)。

建议改为：

```python
pcd_path = sensor_files.get(HESAI_LIDAR_NAME)
```

并校验：

```python
if pcd_path is None:
    raise RuntimeError(
        f"missing lidar sensor {HESAI_LIDAR_NAME}, "
        f"available={list(sensor_files)}"
    )
```

否则将来有多路雷达 PCD 时，“第一个 PCD”可能不是 Pandar P128。

### 3. `config.py`：增加转换配置

建议增加：

```python
MCAP_LIDAR_TOPIC = "/lidar/pandar"
MCAP_LIDAR_SENSOR_NAME = "p128_0"
MCAP_OUTPUT_BINARY = True
MCAP_TIMESTAMP_FIELD = ""
MCAP_RING_FIELD = ""
```

字段名最好允许自动发现后由环境变量覆盖。

### 4. S3 上传工具或数据准备入口

现有推理只接受 S3 路径，所以转换后还需要：

```text
本地 PCD/meta.json → 上传 S3 → 执行 prelabel_main
```

建议新增独立准备入口：

```text
prelabel_model/prepare_local_dataset.py
```

负责：

1. 调用 MCAP 转换；
2. 生成 `meta.json`；
3. 上传整个任务目录；
4. 返回可直接传给 `--param_url` 的路径。

不建议为了这批本地数据，把 `prelabel_main.py` 全面改成同时处理本地路径和 S3。正式流水线保持单一的 S3 输入会更稳定。

## 六、哪些文件不需要修改

- `runtime_bridge.py`：仍然负责加载现有 Hesai 模型。
- `MetaManager`：转换器生成初始 meta；推理节点继续安全更新 meta。
- Kafka 通知模块：无需修改。
- camera MCAP：当前点云模型不使用，暂时不转换也不写入 meta。

## 七、推荐实施顺序

1. 先实现 `--inspect-only`，打印 `/lidar/pandar` 第一帧的所有字段、偏移、数据类型、点数和时间戳。
2. 确认 `/lidar/pandar` 确实对应 P128。
3. 对照模型的 `load_pcd()`，确定准确的 PCD字段名和类型。
4. 只转换 3～5 帧。
5. 用 `module.load_pcd()` 回读生成的 PCD，检查 `timestamps`、`rings` 是否非空。
6. 跑 1 帧模型推理。
7. 再批量转换 6 个 MCAP、生成 meta、上传 S3。
8. 最后优化断点续转和并行写盘。

最合适的改造边界是：

```text
新增：MCAP读取 + PCD写入 + meta生成 + 数据上传
修改：prelabel_main只下载目标雷达
修改：detector按p128_0精确选取PCD
保留：模型推理、MetaManager、Kafka整体逻辑
```

这比直接把 MCAP 转换嵌进推理入口更清晰，也能避免每次重跑模型时重复转换 12 GB 原始数据。
