from __future__ import annotations

"""简单 LRU cache 辅助函数。

ROI pipeline 会重复访问相同 PCD、标定和投影结果，用 OrderedDict 实现
小型缓存即可避免重复 IO 和重复投影计算。
"""

from collections import OrderedDict
from typing import Any


def cache_put(cache: OrderedDict[Any, Any], key: Any, value: Any, max_items: int) -> None:
    """写入缓存并在超过容量时淘汰最旧条目。"""
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_items:
        cache.popitem(last=False)


def get_cached(cache: OrderedDict[Any, Any], key: Any) -> Any | None:
    """读取缓存命中项，并把它移动到最近使用位置。"""
    if key not in cache:
        return None
    cache.move_to_end(key)
    return cache[key]
