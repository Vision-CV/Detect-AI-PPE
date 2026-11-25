"""Области тела человека: где искать каску, где жилет, видна ли часть тела.

Если ключевые точки позы видны — область строится по ним (так она следует
за реальной позой: наклон, присед). Иначе используются доли бокса человека.
"""

from __future__ import annotations

from statistics import mean

from ppe_monitor.domain.geometry import (
    bounding_box,
    box_height,
    box_width,
    distance,
    visible_points,
)
from ppe_monitor.domain.models import Box, Kp, Person
from ppe_monitor.domain.policy import Region


def head_region(person: Person, min_vis: float) -> Box:
    x1, y1, x2, y2 = person.box
    h = box_height(person.box)
    head_pts = visible_points(person.keypoints, Kp.HEAD, min_vis)
    if not head_pts:
        return (x1, y1 - 0.10 * h, x2, y1 + 0.30 * h)

    cx = mean(p[0] for p in head_pts)
    cy = mean(p[1] for p in head_pts)
    # Оценка размера головы: ~1/8 роста, но у присевшего человека рост
    # в кадре меньше — тогда опираемся на ширину плеч.
    size = 0.12 * h
    shoulders = visible_points(person.keypoints, Kp.SHOULDERS, min_vis)
    if len(shoulders) == 2:
        size = max(size, 0.45 * distance(shoulders[0], shoulders[1]))
    # Каска сидит выше глаз, поэтому область вытянута вверх.
    return (cx - 1.3 * size, cy - 1.8 * size, cx + 1.3 * size, cy + 0.7 * size)


def torso_region(person: Person, min_vis: float) -> Box:
    x1, y1, x2, y2 = person.box
    h = box_height(person.box)
    torso_pts = visible_points(person.keypoints, Kp.TORSO, min_vis)
    if len(torso_pts) < 2:
        return (x1, y1 + 0.15 * h, x2, y1 + 0.70 * h)

    bx1, by1, bx2, by2 = bounding_box(torso_pts)
    if not visible_points(person.keypoints, Kp.HIPS, min_vis):
        by2 = by1 + 0.30 * h  # бёдра не видны — достраиваем торс вниз от плеч
    pad_x = max(0.5 * (bx2 - bx1), 0.10 * h)
    pad_y = 0.08 * h
    return (bx1 - pad_x, by1 - pad_y, bx2 + pad_x, by2 + pad_y)


def body_region(person: Person) -> Box:
    x1, y1, x2, y2 = person.box
    pad = 0.05 * box_width(person.box)
    return (x1 - pad, y1 - pad, x2 + pad, y2 + pad)


def region_box(person: Person, region: str, min_vis: float) -> Box:
    if region == Region.HEAD:
        return head_region(person, min_vis)
    if region == Region.TORSO:
        return torso_region(person, min_vis)
    return body_region(person)


def is_region_visible(
    person: Person,
    region: str,
    min_vis: float,
    edge_margin: float,
) -> bool:
    """Можно ли по этому кадру судить о наличии СИЗ на данной части тела.

    Если нельзя — отсутствие СИЗ не считается нарушением (статус UNKNOWN).
    """
    if region == Region.HEAD:
        if person.box[1] <= edge_margin:
            return False  # голова обрезана верхним краем кадра
        if person.keypoints is None:
            return True
        return bool(
            visible_points(person.keypoints, Kp.HEAD, min_vis)
            or len(visible_points(person.keypoints, Kp.SHOULDERS, min_vis)) == 2
        )
    if region == Region.TORSO:
        if person.keypoints is None:
            return True
        return len(visible_points(person.keypoints, Kp.TORSO, min_vis)) >= 2
    return True
