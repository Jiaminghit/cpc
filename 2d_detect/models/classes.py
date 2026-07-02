from __future__ import annotations

"""检测类别表与开放词汇标签归一化。

GroundingDINO 返回的是文本短语，后续评估与 ROI 过滤需要稳定的
`class_name/class_id`。这里集中维护类别表、别名和匹配优先级。
"""

from typing import Optional


# 输出 JSON 中使用的统一类别定义。
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


def build_alias_index() -> list[tuple[str, str]]:
    """将别名表展开为按匹配优先级排序的列表。"""
    items = []
    for class_name, aliases in ALIASES.items():
        for alias in aliases:
            items.append((alias.lower(), class_name))
    return sorted(items, key=lambda item: (-len(item[0]), CLASS_PRIORITY[item[1]]))


def canonicalize_label(raw_phrase: str, alias_index: list[tuple[str, str]]) -> tuple[Optional[str], Optional[int]]:
    """把模型返回的原始短语映射为统一类别；无法识别时返回 `(None, None)`。"""
    normalized = raw_phrase.lower().replace("_", " ").strip()
    for alias, class_name in alias_index:
        if alias in normalized:
            return class_name, CLASS_ID_BY_NAME[class_name]
    return None, None
