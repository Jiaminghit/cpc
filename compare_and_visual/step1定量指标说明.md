# Step 1 定量指标说明

本文档说明 `compare_and_visual/evaluate_roi_projection_vs_dino.py` 输出的第一层 ROI 内 2D 一致性评价指标。

当前评价口径：

```text
3D 投影框 = 被评价对象
DINO ROI 检测框 = 2D baseline / pseudo reference
评价结果 = pseudo consistency，不等价于人工真值结论
```

## 1. 输入规模指标

### aligned_records

参与评价的对齐记录数。

一条 record 对应：

```text
pcd_timestamp + camera
```

例如：

```text
aligned_records: 3000
```

表示共有 3000 张按点云时间戳和相机视角对齐后的图像参与评价。

### projection_boxes

参与评价的 ROI 内 3D 投影框数量。

来源字段：

```text
projections_aligned.jsonl
  -> projected_boxes[]
```

这些框来自 3D 检测框投影到 2D 图像后的 `bbox_xyxy`。

### dino_boxes

参与评价的 ROI 内 GroundingDINO 检测框数量。

来源字段：

```text
detections_aligned_roi_only.jsonl
  -> detections[]
```

这些框作为当前第一层评价中的 2D baseline。

## 2. 匹配状态指标

评价脚本会在每个 `(pcd_timestamp, camera)` 内，将：

```text
projected_boxes[].bbox_xyxy
```

和：

```text
detections[].bbox_xyxy
```

做 IoU 一对一匹配。

默认阈值：

```text
iou_threshold = 0.5
low_iou_threshold = 0.3
```

### tp

几何和类别都一致的匹配数量。

判定条件：

```text
IoU >= iou_threshold
and projection.class_name == dino.class_name
```

默认即：

```text
IoU >= 0.5
and class_name 相同
```

含义：

```text
3D 投影框和 DINO ROI 框在图像上位置吻合，类别也一致。
```

这是当前最可靠的一致性结果。

### class_mismatch

几何匹配成功，但类别不一致的数量。

判定条件：

```text
IoU >= iou_threshold
and projection.class_name != dino.class_name
```

默认即：

```text
IoU >= 0.5
and class_name 不同
```

含义：

```text
两个框在 2D 图像上对得上，但 3D 类别和 DINO 类别不同。
```

后续可映射为：

```text
类别错误疑似
```

但仍需人工或更多规则确认，因为 DINO 本身也可能分类错误。

### low_iou_matched

有一定重叠但贴合不够好的匹配数量。

判定条件：

```text
low_iou_threshold <= IoU < iou_threshold
```

默认即：

```text
0.3 <= IoU < 0.5
```

含义：

```text
3D 投影框和 DINO 框有明显重叠，但 2D 贴合程度不足。
```

后续可映射为：

```text
贴合错误疑似
```

### weak_overlap_matched

只有弱重叠的匹配数量。

判定条件：

```text
0 < IoU < low_iou_threshold
```

默认即：

```text
0 < IoU < 0.3
```

含义：

```text
两个框只有少量交集，被贪心匹配策略配成了一对。
```

这类结果需要谨慎解读，可能代表：

```text
1. 3D 投影框和 DINO 框贴合很差。
2. 图像中目标密集，贪心匹配找到了弱相关框。
3. 3D 投影、DINO 检测、ROI 估计或时间同步存在偏差。
```

### unmatched_projection

未匹配到 DINO ROI 框的 3D 投影框数量。

判定条件：

```text
某个 projected_box 没有被分配到任何 dino_box
```

含义：

```text
3D 投影认为 ROI 内有目标，但 DINO baseline 没有对应的 2D 检测框。
```

可能解释：

```text
1. 3D 多标疑似。
2. DINO baseline 漏检。
3. 目标在图像中被遮挡或很难识别。
4. 投影框与图像目标存在错位。
5. ROI 或时间同步误差影响了对齐。
```

该指标不能直接等价为“3D 错误”，只能表示“3D 投影与 DINO baseline 不一致”。

### unmatched_dino

未匹配到 3D 投影框的 DINO ROI 框数量。

判定条件：

```text
某个 dino_box 没有被分配到任何 projected_box
```

含义：

```text
DINO baseline 在 ROI 内检测到了目标，但 3D 投影结果中没有对应框。
```

可能解释：

```text
1. 3D 漏标疑似。
2. DINO baseline 误检。
3. DINO 框的 LiDAR ROI 估计有偏差。
4. 类别、尺度、投影或时间同步导致无法匹配。
```

该指标也不能直接等价为“3D 漏标”，只能作为疑似问题入口。

## 3. 核心公式

### geometry_tp

几何匹配成功数量。

公式：

```text
geometry_tp = tp + class_mismatch
```

原因：

```text
class_mismatch 虽然类别不一致，但 IoU >= iou_threshold，
因此在几何位置上仍然算作匹配成功。
```

### pseudo_precision

伪精度。

公式：

```text
pseudo_precision = geometry_tp / (geometry_tp + unmatched_projection)
```

展开：

```text
pseudo_precision = (tp + class_mismatch)
                 / (tp + class_mismatch + unmatched_projection)
```

含义：

```text
以 DINO ROI baseline 为参考，有多少 3D 投影框能够找到高 IoU 对应框。
```

注意：

```text
这是 pseudo precision，不是人工真值 precision。
```

### pseudo_recall

伪召回率。

公式：

```text
pseudo_recall = geometry_tp / (geometry_tp + unmatched_dino)
```

展开：

```text
pseudo_recall = (tp + class_mismatch)
              / (tp + class_mismatch + unmatched_dino)
```

含义：

```text
以 DINO ROI baseline 为参考，有多少 DINO ROI 框被 3D 投影框高 IoU 覆盖到。
```

注意：

```text
这是 pseudo recall，不是人工真值 recall。
```

### pseudo_f1

伪 F1。

公式：

```text
pseudo_f1 = 2 * pseudo_precision * pseudo_recall
            / (pseudo_precision + pseudo_recall)
```

含义：

```text
综合 pseudo_precision 和 pseudo_recall 的整体一致性指标。
```

当 `pseudo_precision + pseudo_recall == 0` 时：

```text
pseudo_f1 = null
```

## 4. 示例计算

假设全量结果为：

```text
tp: 1320
class_mismatch: 76
unmatched_projection: 9142
unmatched_dino: 4658
```

则：

```text
geometry_tp = tp + class_mismatch
            = 1320 + 76
            = 1396
```

伪精度：

```text
pseudo_precision = 1396 / (1396 + 9142)
                 = 1396 / 10538
                 ≈ 0.132473
```

伪召回率：

```text
pseudo_recall = 1396 / (1396 + 4658)
              = 1396 / 6054
              ≈ 0.230591
```

伪 F1：

```text
pseudo_f1 = 2 * 0.132473 * 0.230591
            / (0.132473 + 0.230591)
          ≈ 0.168274
```

## 5. IoU 与辅助几何指标

### IoU

两个 2D bbox 的交并比。

公式：

```text
IoU = area(intersection) / area(union)
```

其中：

```text
union = area(projection_box) + area(dino_box) - area(intersection)
```

IoU 越高，两个框越贴合。

### mean_iou_matched

所有已匹配 pair 的平均 IoU。

统计范围包括：

```text
tp
class_mismatch
low_iou_matched
weak_overlap_matched
```

不包括：

```text
unmatched_projection
unmatched_dino
```

### median_iou_matched

所有已匹配 pair 的 IoU 中位数。

相比平均值，中位数对极端低 IoU 或极端高 IoU 更不敏感。

### mean_center_distance_px

所有已匹配 pair 的 2D bbox 中心点平均距离，单位是像素。

中心点计算：

```text
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
```

中心距离：

```text
center_distance_px =
  sqrt((projection_cx - dino_cx)^2 + (projection_cy - dino_cy)^2)
```

含义：

```text
数值越小，两个框中心越接近。
```

## 6. warnings

### invalid projection bbox

表示某些 3D 投影框的 `bbox_xyxy` 无法用于 IoU 计算。

常见原因：

```text
1. bbox 长度不是 4。
2. bbox 中存在非数值或非有限数值。
3. x2 <= x1。
4. y2 <= y1。
5. bbox clip 到图像边界后面积为 0。
```

处理方式：

```text
脚本会跳过这些无效 bbox，并将对应 pcd_timestamp、camera、box index 写入 warnings。
```

如果数量很少，例如：

```text
warnings: 5 个 invalid projection bbox
```

说明它们对整体统计影响较小，但仍建议后续单独回查。

## 7. 解读注意事项

### 不能把 DINO 当作人工真值

当前所有 precision / recall / F1 都带有 `pseudo` 前缀，因为 DINO 只是 baseline，不是人工标注真值。

因此：

```text
unmatched_projection 高
```

不一定说明 3D 一定多标，也可能是 DINO 漏检。

同理：

```text
unmatched_dino 高
```

不一定说明 3D 一定漏标，也可能是 DINO 误检。

### class_mismatch 需要单独看

`class_mismatch` 在几何上算匹配成功，但在类别上存在冲突。

后续分析时应查看：

```text
class_mismatch_pairs
```

例如：

```text
Car->Van
Van->Car
Pedestrian->Cyclist
```

这有助于发现高频类别混淆。

### low_iou 和 weak_overlap 是贴合问题入口

这两类不是严格 TP，也不是完全 unmatched。

它们通常适合进入后续可视化抽查：

```text
low_iou_matched
weak_overlap_matched
```

重点看：

```text
1. 投影框是否整体偏移。
2. 框尺寸是否明显过大或过小。
3. 是否存在相机标定或时间同步问题。
4. 是否有遮挡导致 DINO 框只覆盖局部目标。
```

## 8. 当前示例结果一句话解读

示例结果：

```text
projection_boxes: 13416
dino_boxes: 8932
tp: 1320
class_mismatch: 76
low_iou_matched: 1396
weak_overlap_matched: 1482
unmatched_projection: 9142
unmatched_dino: 4658
pseudo_precision: 0.132473
pseudo_recall: 0.230591
pseudo_f1: 0.168274
```

说明：

```text
当前 ROI 内 3D 投影框与 DINO ROI baseline 的 2D 一致性偏低。
主要不一致来源是大量 unmatched_projection 和 unmatched_dino。
下一步应抽样可视化这些 unmatched 与 low_iou 结果，
判断问题来自 3D 标注、DINO baseline、投影链路、ROI 过滤还是时间同步。
```
