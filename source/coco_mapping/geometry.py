from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def bbox_xywh_from_polygon_rel(
    x_rel: float,
    y_rel: float,
    polygon_rel: Sequence[Sequence[float]],
    width: int,
    height: int,
    buffer: float = 0.05,
) -> list[int]:
    """
    Build a COCO xywh bbox in pixels from a polygon whose vertices are relative offsets
    around an anchor point (x_rel, y_rel), as Squidle stores them.
    """
    min_x = (np.min([p[0] for p in polygon_rel]) + x_rel) * width
    max_x = (np.max([p[0] for p in polygon_rel]) + x_rel) * width
    min_y = (np.min([p[1] for p in polygon_rel]) + y_rel) * height
    max_y = (np.max([p[1] for p in polygon_rel]) + y_rel) * height

    width_buffer = (max_x - min_x) * buffer
    height_buffer = (max_y - min_y) * buffer

    x0 = int(min_x - width_buffer)
    y0 = int(min_y - height_buffer)
    x1 = int(max_x + width_buffer)
    y1 = int(max_y + height_buffer)
    return clip_bbox_xywh([x0, y0, x1 - x0, y1 - y0], width=width, height=height)


def bbox_xywh_from_point_rel(
    x_rel: float,
    y_rel: float,
    width: int,
    height: int,
    buffer: float = 0.035,
) -> list[int]:
    """
    Estimate a COCO xywh bbox in pixels around a relative point.
    """
    x0 = int((x_rel - buffer) * width)
    x1 = int((x_rel + buffer) * width)
    y0 = int((y_rel - buffer) * height)
    y1 = int((y_rel + buffer) * height)
    return clip_bbox_xywh([x0, y0, x1 - x0, y1 - y0], width=width, height=height)


def clip_bbox_xywh(bbox_xywh: Sequence[int], width: int, height: int) -> list[int]:
    x, y, w, h = [int(v) for v in bbox_xywh]
    if w <= 0 or h <= 0:
        return [0, 0, 0, 0]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    w2 = max(0, x1 - x0)
    h2 = max(0, y1 - y0)
    return [x0, y0, w2, h2]


def polygon_px_from_rel(
    x_rel: float, y_rel: float, polygon_rel: Sequence[Sequence[float]], width: int, height: int
) -> list[list[int]]:
    return [[int((p[0] + x_rel) * width), int((p[1] + y_rel) * height)] for p in polygon_rel]


def is_valid_polygon(polygon_px: Iterable[Sequence[int]] | None) -> bool:
    if not polygon_px:
        return False
    return sum(1 for _ in polygon_px) >= 3

