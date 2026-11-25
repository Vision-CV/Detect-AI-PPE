"""Конструкторы тестовых данных: человек с правдоподобной позой, СИЗ на нём."""

from __future__ import annotations

from ppe_monitor.domain.models import Keypoint, Person, PPEItem

VISIBLE = 0.9
HIDDEN = 0.0


def make_keypoints(
    box: tuple[float, float, float, float],
    *,
    head: float = VISIBLE,
    torso: float = VISIBLE,
    ankles: float = VISIBLE,
) -> tuple[Keypoint, ...]:
    """17 точек COCO для стоящего человека в боксе. Видимость задаётся группами."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    kps: list[Keypoint] = [(cx, y1 + 0.5 * h, HIDDEN)] * 17
    head_y = y1 + 0.08 * h
    for idx, dx in ((0, 0.0), (1, -0.04), (2, 0.04), (3, -0.08), (4, 0.08)):
        kps[idx] = (cx + dx * w, head_y, head)
    for idx, dx in ((5, -0.25), (6, 0.25)):
        kps[idx] = (cx + dx * w, y1 + 0.2 * h, torso)
    for idx, dx in ((11, -0.15), (12, 0.15)):
        kps[idx] = (cx + dx * w, y1 + 0.52 * h, torso)
    for idx, dx in ((15, -0.1), (16, 0.1)):
        kps[idx] = (cx + dx * w, y1 + 0.98 * h, ankles)
    return tuple(kps)


def person(
    track_id: int = 1,
    box: tuple[float, float, float, float] = (100, 100, 200, 400),
    *,
    keypoints: bool = True,
    **visibility: float,
) -> Person:
    return Person(
        track_id=track_id,
        box=box,
        confidence=0.9,
        keypoints=make_keypoints(box, **visibility) if keypoints else None,
    )


def helmet_on(p: Person, conf: float = 0.8) -> PPEItem:
    x1, y1, x2, y2 = p.box
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    return PPEItem("helmet", (cx - 0.2 * w, y1 - 0.02 * h, cx + 0.2 * w, y1 + 0.08 * h), conf)


def vest_on(p: Person, conf: float = 0.8) -> PPEItem:
    x1, y1, x2, y2 = p.box
    w, h = x2 - x1, y2 - y1
    return PPEItem("vest", (x1 + 0.1 * w, y1 + 0.2 * h, x2 - 0.1 * w, y1 + 0.55 * h), conf)


def helmet_in_hand(p: Person, conf: float = 0.8) -> PPEItem:
    """Каска в опущенной руке: внутри бокса человека, но на уровне пояса."""
    x1, y1, x2, y2 = p.box
    h = y2 - y1
    return PPEItem("helmet", (x2 - 30, y1 + 0.5 * h, x2 + 5, y1 + 0.6 * h), conf)
