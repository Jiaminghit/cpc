from __future__ import annotations

from typing import Any

import numpy as np


def clean_bbox(bbox: list[Any], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = int(round(np.clip(x1, 0, width - 1)))
    y1 = int(round(np.clip(y1, 0, height - 1)))
    x2 = int(round(np.clip(x2, 0, width - 1)))
    y2 = int(round(np.clip(y2, 0, height - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2
