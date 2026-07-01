"""类别体系、类别颜色和标签归一化工具。

任务角色：
    管理项目中 3D 检测框类别的标准名称、类别 ID、可视化颜色和
    原始模型标签到标准类别的映射规则，确保所有投影脚本使用同一套类别定义。
"""

from __future__ import annotations


CLASS_TABLE = [
    {"class_id": 0, "class_name": "Car"},
    {"class_id": 1, "class_name": "Pedestrian"},
    {"class_id": 2, "class_name": "Cyclist"},
    {"class_id": 3, "class_name": "Van"},
    {"class_id": 4, "class_name": "Traffic_cone"},
]
CLASS_ID_BY_NAME = {item["class_name"]: item["class_id"] for item in CLASS_TABLE}
ALIASES = {
    "Car": ["car", "vehicle"],
    "Pedestrian": ["pedestrian", "person"],
    "Cyclist": ["person riding bicycle", "person riding bike", "bicyclist", "cyclist"],
    "Van": ["minivan", "van"],
    "Traffic_cone": ["traffic cone", "traffic_cone", "cone"],
}
CLASS_PRIORITY = {
    "Traffic_cone": 0,
    "Cyclist": 1,
    "Van": 2,
    "Pedestrian": 3,
    "Car": 4,
}
CLASS_COLORS = {
    "Car": (255, 80, 40),
    "Van": (255, 220, 40),
    "Pedestrian": (60, 220, 60),
    "Cyclist": (40, 220, 255),
    "Traffic_cone": (40, 40, 255),
}
DEFAULT_COLOR = (220, 220, 220)


# 工作角色：构建原始标签别名到标准类别名的查找表。
# 输入输出：无显式输入，读取模块内 ALIASES / CLASS_PRIORITY，输出按匹配优先级排序的别名列表。
# 实现思路：长别名优先，类别优先级次之，避免 "traffic cone" 被 "car" 等短词误匹配。
def build_alias_index() -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for class_name, aliases in ALIASES.items():
        for alias in aliases:
            items.append((alias.lower(), class_name))
    return sorted(items, key=lambda item: (-len(item[0]), CLASS_PRIORITY[item[1]]))


ALIAS_INDEX = build_alias_index()


# 工作角色：把模型输出的原始类别文本归一化为项目标准类别。
# 输入输出：输入 raw_label 字符串，输出 (class_name, class_id)，无法识别时输出 (None, None)。
# 实现思路：先统一大小写和下划线，再按 ALIAS_INDEX 逐项子串匹配。
def canonicalize_label(raw_label: str) -> tuple[str | None, int | None]:
    normalized = raw_label.lower().replace("_", " ").strip()
    for alias, class_name in ALIAS_INDEX:
        if alias in normalized:
            return class_name, CLASS_ID_BY_NAME[class_name]
    return None, None
