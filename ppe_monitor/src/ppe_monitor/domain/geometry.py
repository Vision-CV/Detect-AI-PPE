"""Геометрические функции без внешних зависимостей."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from ppe_monitor.domain.models import Box, Keypoint, Point


def box_width(box: Box) -> float:
    return box[2] - box[0]


def box_height(box: Box) -> float:
    return box[3] - box[1]


def box_center(box: Box) -> Point:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def point_in_box(point: Point, box: Box) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def iou(a: Box, b: Box) -> float:
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    union = box_width(a) * box_height(a) + box_width(b) * box_height(b) - inter
    return inter / union


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Метод трассировки луча: считаем пересечения горизонтального луча
    из точки с рёбрами многоугольника. Нечётное число — точка внутри."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def visible_points(
    keypoints: Sequence[Keypoint] | None,
    indices: Iterable[int],
    min_visibility: float,
) -> list[Point]:
    """Координаты ключевых точек из `indices`, видимость которых не ниже порога."""
    if keypoints is None:
        return []
    result = []
    for idx in indices:
        if idx < len(keypoints):
            x, y, vis = keypoints[idx]
            if vis >= min_visibility:
                result.append((x, y))
    return result


def bounding_box(points: Sequence[Point]) -> Box:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))
