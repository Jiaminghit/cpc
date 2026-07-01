# Data Convert

本目录提供三类 MCAP 数据转换工具：

- lidar MCAP -> PCD 点云文件
- camera MCAP -> JPG 图像文件
- struct MCAP -> JSON 标定文件

建议在已经配置好的 `mcap_convert` 环境中运行：

```bash
conda activate mcap_convert
cd /home/c64508/桌面/dataset
```

推荐直接使用顶层统一入口：

```bash
python -m data_convert.convert_dataset \
  /home/c64508/桌面/dataset/2067268107790897153 \
  --overwrite
```

顶层入口默认只转换 6 路常用 camera topic：

```text
/sensor/camera_front_wide_image
/sensor/camera_rear_image
/sensor/camera_side_left_front_image
/sensor/camera_side_left_rear_image
/sensor/camera_side_right_front_image
/sensor/camera_side_right_rear_image
```

如需处理 MCAP 中的全部 ImageV2 camera topic，可在顶层入口加：

```bash
--all-camera-topics
```

底层转换实现集中放在 `data_convert/util/` 下；需要单独调试某一步时，也可以直接运行对应的 `data_convert.util.*` 模块。

## 第一部分：用户说明

### 1. MCAP 转 PCD

`util/mcap_to_pcd.py` 用于将 lidar MCAP 中的 `PointCloud2V2` 消息转换为二进制 PCD 文件。

默认读取 topic：

```text
/lidar/pandar
```

数据流：

```text
.mcap
  -> mcap_reader.iter_pointcloud_frames()
  -> PointCloudFrame
  -> pcd_writer.write_pcd()
  -> <output>/<sensor_name>/<timestamp>.pcd
  -> meta.json
```

常用命令：

```bash
python -m data_convert.util.mcap_to_pcd \
  --input-dir /home/c64508/桌面/dataset/2067268107790897153/lidar \
  --output-dir /home/c64508/桌面/dataset/2067268107790897153/pcd \
  --topic /lidar/pandar \
  --sensor-name p128_0 \
  --overwrite
```

查看 MCAP 内部 topic：

```bash
python -m data_convert.util.mcap_to_pcd \
  --input-file /path/to/lidar.mcap \
  --inspect-only
```

测试前几帧：

```bash
python -m data_convert.util.mcap_to_pcd \
  --input-file /path/to/lidar.mcap \
  --output-dir /tmp/pcd_check \
  --topic /lidar/pandar \
  --sensor-name p128_0 \
  --max-frames 3 \
  --overwrite
```

输出结构：

```text
output_dir/
  meta.json
  p128_0/
    1781707303798187008.pcd
    1781707303898187008.pcd
    ...
```

`meta.json` 中会记录：

- `dataset_type_code = "lidar"`
- sensor 名称和原始 topic
- 每一帧 timestamp
- 每个 timestamp 对应的 PCD 相对路径
- 转换来源和点数信息

### 2. MCAP 转 JPG

`util/mcap_to_jpg.py` 用于将 camera MCAP 中的 `ImageV2` 消息转换为 JPG 文件。

当前数据中的 camera 图像编码通常是：

```text
encoding = h265
```

H.265 是连续视频码流，不是每条 message 都能独立解码。因此程序会按 topic 将多个 message 拼接为连续 H.265 流，再用 OpenCV/FFmpeg backend 解码为 JPG。

数据流：

```text
.mcap
  -> mcap_reader.iter_image_frames()
  -> ImageFrame sequence
  -> jpg_writer.write_jpg_sequence()
  -> <output>/<sensor_name>/<timestamp>.jpg
  -> meta.json
```

不指定 topic 时，程序会从 MCAP summary 中自动发现所有 `ImageV2` topic：

```bash
python -m data_convert.util.mcap_to_jpg \
  --input-dir /home/c64508/桌面/dataset/2067268107790897153/camera \
  --output-dir /home/c64508/桌面/dataset/2067268107790897153/jpg \
  --overwrite
```

指定一个或多个 topic：

```bash
python -m data_convert.util.mcap_to_jpg \
  --input-file /home/c64508/桌面/dataset/2067268107790897153/camera/video_202606171441_005_0.mcap \
  --output-dir /tmp/camera_jpg \
  --topic front_long=/sensor/camera_front_long_image \
  --topic front_wide=/sensor/camera_front_wide_image \
  --overwrite
```

`--topic` 支持两种格式：

```text
/sensor/camera_front_long_image
front_long=/sensor/camera_front_long_image
```

如果不写 sensor 名，程序会根据 topic 自动生成目录名，例如：

```text
/sensor/camera_front_long_image -> camera_front_long
```

输出结构：

```text
output_dir/
  meta.json
  camera_front_long/
    1781707260228409572.jpg
    1781707260255469568.jpg
    ...
  camera_front_wide/
    ...
```

`meta.json` 中会记录：

- `dataset_type_code = "camera"`
- 每个 camera sensor 的 source topic
- 每个 sensor 的 `input_messages` 和 `decoded_frames`
- 每张 JPG 的 timestamp、相对路径、宽高和原始编码

说明：

- H.265 如果从非关键帧开始，底层解码器可能丢掉开头少量帧。
- 程序默认隐藏 FFmpeg/OpenCV 的 H.265 warning。
- 如需调试底层解码日志，可加：

```bash
--show-decoder-log
```

### 3. MCAP 转 Struct JSON

`util/mcap_to_struct.py` 用于将 struct MCAP 中的 `SensorCalibration` 消息转换为 JSON 标定文件。

默认读取 topic：

```text
/calib/calib_param
```

数据流：

```text
.mcap
  -> mcap_reader.iter_calibration_messages()
  -> CalibrationMessage
  -> struct_writer.write_calibration_json()
  -> <output>/calib/<timestamp>.json
  -> calibration_latest.json
  -> meta.json
```

常用命令：

```bash
python -m data_convert.util.mcap_to_struct \
  --input-dir /home/c64508/桌面/dataset/2067268107790897153/struct \
  --output-dir /home/c64508/桌面/dataset/2067268107790897153/struct_json \
  --overwrite
```

测试前几条 calibration：

```bash
python -m data_convert.util.mcap_to_struct \
  --input-file /home/c64508/桌面/dataset/2067268107790897153/struct/struct_202606171437_001_0.mcap \
  --output-dir /tmp/struct_check \
  --max-frames 3 \
  --overwrite
```

输出结构：

```text
output_dir/
  calib/
    1781707025355076455.json
    1781707026355075388.json
    ...
  calibration_latest.json
  meta.json
```

`calib/<timestamp>.json` 中会包含：

- 当前 calibration message 的 timestamp
- 来源 MCAP 和 topic
- 车辆信息和 calibration version
- sensor 列表
- 每个 sensor 的类型、ID、内参、外参
- 完整 `raw` 字段，避免原始信息丢失

`calibration_latest.json` 保存最后一条 calibration，方便只需要最新标定的流程直接读取。

`meta.json` 中会记录：

- `dataset_type_code = "struct"`
- 每条 calibration timestamp 和文件路径
- latest timestamp 和 latest 文件路径
- 最新一条 calibration 中的 sensor 摘要

## 第二部分：开发说明

### 1. 整体流水线

三个转换任务共享同一套分层方式：

```text
CLI converter
  -> mcap_reader typed iterator
  -> writer
  -> timestamp-named output files
  -> meta.json
```

对应关系：

```text
util/mcap_to_pcd.py
  -> mcap_reader.iter_pointcloud_frames()
  -> pcd_writer.write_pcd()

util/mcap_to_jpg.py
  -> mcap_reader.iter_image_frames()
  -> jpg_writer.write_jpg_sequence()

util/mcap_to_struct.py
  -> mcap_reader.iter_calibration_messages()
  -> struct_writer.write_calibration_json()
```

CLI 脚本负责：

- 解析命令行参数
- 发现输入 MCAP 文件
- 支持 `--inspect-only`
- 调用 reader 读取 typed message
- 调用 writer 写目标文件
- 生成 `meta.json`

Writer 脚本负责：

- 将 typed message 转换为目标格式
- 校验目标格式所需字段
- 原子写入文件
- 返回写入结果，供 CLI 构建 metadata

Reader 脚本负责：

- 统一加载 MCAP 依赖
- 读取 MCAP summary
- 读取 ROS2 message
- 提取 timestamp
- 将 ROS dynamic message 适配为内部 dataclass

### 2. util/mcap_reader.py

`util/mcap_reader.py` 是所有转换任务的读取层。

输入：

```text
path: .mcap 文件路径
topic/topics: 需要读取的 ROS2 topic
```

输出：

- `McapInfo`
- `RosMessageEnvelope`
- `PointCloudFrame`
- `ImageFrame`
- `CalibrationMessage`

重要 dataclass：

`PointField`

- 描述 PointCloud2 中单个字段
- 字段包括 `name/offset/datatype/count`
- 提供 `size/pcd_size/pcd_type`

`PointCloudFrame`

- 一帧点云消息
- 包含 timestamp、topic、width、height、fields、point_step、row_step、data 等
- `util/pcd_writer.py` 直接依赖此结构

`ImageFrame`

- 一条 ImageV2 消息
- 包含 timestamp、topic、width、height、encoding、step、data 等
- `util/jpg_writer.py` 直接依赖此结构

`CalibrationMessage`

- 一条 SensorCalibration 消息
- 包含 timestamp、vehicle 信息、calibration version、sensors 和原始 `ros_msg`
- `util/struct_writer.py` 直接依赖此结构

`TopicInfo` 和 `McapInfo`

- 用于 `--inspect-only`
- 保存 topic、schema、encoding、message count 和 MCAP 时间范围

`RosMessageEnvelope`

- 通用 ROS2 message 包装
- 保存 timestamp、log time、publish time、topic、schema、encoding、ros_msg 和 source file

重要函数：

`_require_mcap()`

- 延迟 import `mcap` 和 `mcap_ros2`
- 缺少依赖时抛出清晰错误

`inspect_mcap(path)`

- 读取 MCAP summary
- 输出 topic/schema/message count
- 不解码 message payload

`iter_ros2_messages(path, topics=None)`

- 通用 message 迭代器
- 返回 `RosMessageEnvelope`
- 新增消息类型时优先复用此函数

`iter_pointcloud_frames(path, topic="/lidar/pandar")`

- 读取 PointCloud2-like message
- 校验点云字段布局
- 输出 `PointCloudFrame`

`iter_image_frames(path, topic)`

- 读取 ImageV2-like message
- 支持 raw image 和 H.265 这类压缩流
- 输出 `ImageFrame`

`iter_calibration_messages(path, topic="/calib/calib_param")`

- 读取 SensorCalibration-like message
- 输出 `CalibrationMessage`

后续添加新类型时建议：

1. 先用 `inspect_mcap()` 确认 topic/schema。
2. 用 `iter_ros2_messages()` 取一条 raw `ros_msg` 查看字段。
3. 在 `util/mcap_reader.py` 中新增对应 dataclass 和 `iter_xxx_messages()`。
4. 新增 `util/mcap_to_xxx.py` 和 `util/xxx_writer.py`。

### 3. util/mcap_to_pcd.py

职责：

- lidar 转换 CLI
- 批量发现 MCAP 文件
- 调用 `iter_pointcloud_frames()`
- 调用 `write_pcd()`
- 生成 lidar `meta.json`

重要类：

`ConvertedFrame`

- 保存转换后的单帧索引信息
- 字段包括 timestamp、sensor name、relative path、source MCAP、source topic、point count

重要函数：

`discover_mcap_files(input_dir, input_files)`

- 合并目录输入和文件输入
- 去重并排序
- 校验 `.mcap` 后缀

`print_inspection(paths)`

- 打印每个 MCAP 的 summary

`convert_mcaps(paths, output_dir, topic, sensor_name, max_frames, overwrite)`

- 主转换函数
- 输出 `list[ConvertedFrame]`

`build_meta(dataset_name, sensor_name, topic, frames)`

- 构建 lidar `meta.json`

### 4. util/pcd_writer.py

职责：

- 将 `PointCloudFrame` 写为 binary PCD
- 保留原始点字段顺序、类型、count
- 处理 PointCloud2 row padding 和 field padding

重要函数：

`write_pcd(output_path, frame, overwrite=False)`

- 写单帧 PCD
- 原子写入
- 不支持 big-endian 点云

`_pcd_header(frame, fields)`

- 生成 PCD v0.7 header

`_is_tightly_packed(frame, fields)`

- 判断 point data 是否可直接写入

`_write_repacked_points(output, frame, fields)`

- 移除 padding 后重新写点数据

### 5. util/mcap_to_jpg.py

职责：

- camera 转换 CLI
- 支持自动发现 ImageV2 topic
- 支持一个或多个 `--topic`
- 每个 topic 输出到独立 sensor 文件夹
- 生成 camera `meta.json`

重要类：

`TopicSpec`

- `sensor_name`
- `topic`

`ConvertedImage`

- 保存单张 JPG 的索引信息
- 包括 timestamp、sensor、relative path、source topic、width、height、encoding

`SensorConversionStats`

- 保存每个 sensor 的 input message 数和 decoded JPG 数

重要函数：

`sensor_name_from_topic(topic)`

- 根据 topic 生成默认 sensor 文件夹名

`parse_topic_specs(raw_topics)`

- 解析 `/topic` 或 `sensor=/topic`

`discover_image_topic_specs(paths)`

- 从 MCAP summary 中自动发现 `ImageV2` topic

`convert_mcaps(paths, output_dir, topic_specs, max_frames, overwrite, quality, show_decoder_log)`

- 主转换函数
- 每个 sensor 独立统计输入和输出

`build_meta(dataset_name, topic_specs, frames, stats)`

- 构建 camera `meta.json`

### 6. util/jpg_writer.py

职责：

- 将 `ImageFrame` 写为 JPG
- 支持 H.265 连续流
- 支持 JPEG/PNG/raw RGB/BGR/mono/BGRA/RGBA

重要类：

`WrittenJpg`

- 保存写出的 JPG 信息
- 字段包括 timestamp、path、width、height、source encoding、source message index

重要函数：

`write_jpg(output_path, frame, overwrite=False, quality=95)`

- 写单个非 H.265 frame
- JPEG/PNG 使用 `cv2.imdecode`
- raw image 使用 numpy reshape 和必要的颜色转换

`write_jpg_sequence(output_dir, frames, overwrite=False, quality=95, used_timestamps=None, show_decoder_log=False)`

- 写一组 ImageFrame
- H.265 会拼接为临时 `.h265` 文件，再用 `cv2.VideoCapture` 解码
- 非 H.265 会逐帧调用 `write_jpg`

`_suppress_stderr()`

- 默认隐藏 OpenCV/FFmpeg 的 H.265 warning
- CLI 中可通过 `--show-decoder-log` 打开底层日志

### 7. util/mcap_to_struct.py

职责：

- struct 转换 CLI
- 批量读取 `/calib/calib_param`
- 每条 message 输出 `calib/<timestamp>.json`
- 输出 `calibration_latest.json`
- 生成 struct `meta.json`

重要类：

`ConvertedCalibration`

- 保存单条 calibration JSON 的索引信息
- 字段包括 timestamp、relative path、source MCAP、source topic、sensor count、calibration version

重要函数：

`convert_mcaps(paths, output_dir, topic="/calib/calib_param", max_frames=None, overwrite=False)`

- 主转换函数
- 输出 `list[ConvertedCalibration]` 和 latest `CalibrationMessage`

`build_meta(dataset_name, topic, frames, latest_message)`

- 构建 struct `meta.json`
- `frame_map` 中每个 timestamp 指向一个 calibration JSON
- `metadata.conversion.latest_file` 指向 `calibration_latest.json`

`_sensor_summary(message)`

- 从最新 calibration 中抽取 sensor 摘要
- 写入 `meta.json` 的 `sensors` 字段

### 8. util/struct_writer.py

职责：

- 将 `CalibrationMessage` 转为 JSON dict
- 提取 sensor 内外参
- 保留完整 raw 字段

重要类：

`WrittenCalibration`

- 保存写出的 calibration JSON 信息
- 字段包括 timestamp、path、sensor count

重要函数：

`to_jsonable(value)`

- 递归将 ROS dynamic object 转换为 JSON 可序列化对象
- 支持 primitive、list、tuple、dict、bytes 和动态对象属性

`extract_extrinsic(sensor)`

- 提取外参：
  - `translation`
  - `rotation`
  - `transform_matrix`
  - `transform_matrix_4x4`

`extract_intrinsics(sensor)`

- 提取相机内参：
  - `width`
  - `height`
  - `fx/fy/cx/cy`
  - `camera_matrix`
  - `distortion_model`
  - `distortion_coeffs`
  - `raw`

`extract_sensor(sensor)`

- 转换单个 `SensorCalibItem`
- 标准化输出 `name/type/sensor_type_code/intrinsics/extrinsic/raw`

`calibration_to_dict(message)`

- 转换整条 `CalibrationMessage`
- 输出最终 JSON 内容

`write_calibration_json(output_path, message, overwrite=False)`

- 写单条 calibration JSON
- 原子写入

### 9. 维护建议

新增转换任务时，优先保持下面的文件结构：

```text
util/mcap_reader.py
  -> 新增 dataclass
  -> 新增 iter_xxx_messages()

util/mcap_to_xxx.py
  -> CLI、批处理、meta.json

util/xxx_writer.py
  -> 目标格式写入
```

新增输出格式时，建议保持：

- timestamp 作为文件名
- `meta.json` 作为索引
- `--inspect-only`
- `--max-frames`
- `--overwrite`
- 原子写入
- 保留 source MCAP、source topic、timestamp 信息

这样 lidar、camera、struct 以及后续新增任务可以保持同一套使用方式和维护方式。
