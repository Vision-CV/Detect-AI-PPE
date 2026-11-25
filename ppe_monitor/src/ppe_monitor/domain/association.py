"""Привязка найденных СИЗ к людям.

Алгоритм:
1. Для каждой пары (СИЗ, человек) проверяем, что центр СИЗ лежит в нужной
   области тела (каска — голова, жилет — торс) и что размер правдоподобен.
2. Оценка пары = близость к центру области × уверенность детектора.
3. Жадное сопоставление по убыванию оценки: каждый объект СИЗ достаётся
   максимум одному человеку, и у человека не больше `max_per_person` штук
   одного класса. Так человек на переднем плане не «забирает» каску
   стоящего позади.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

from ppe_monitor.domain.body import region_box
from ppe_monitor.domain.geometry import box_center, box_width, distance, point_in_box
from ppe_monitor.domain.models import Box, Person, PPEItem
from ppe_monitor.domain.policy import PPEPolicy


def _closeness(point: tuple[float, float], region: Box) -> float:
    """1.0 в центре области, ~0.0 в её углу."""
    center = box_center(region)
    half_diag = distance(center, (region[2], region[3]))
    if half_diag == 0:
        return 0.0
    return max(0.0, 1.0 - distance(point, center) / half_diag)


def associate(
    persons: Sequence[Person],
    ppe_items: Sequence[PPEItem],
    policy: PPEPolicy,
) -> list[Person]:
    """Возвращает НОВЫЕ объекты Person с заполненным `ppe` (входные не меняются)."""
    min_vis = policy.min_keypoint_visibility
    candidates: list[tuple[float, int, int]] = []

    for item_idx, item in enumerate(ppe_items):
        center = box_center(item.box)
        region_name = policy.region_of(item.class_name)
        max_ratio = policy.max_width_ratio.get(item.class_name)

        for person_idx, person in enumerate(persons):
            if max_ratio is not None and box_width(item.box) > max_ratio * box_width(person.box):
                continue
            region = region_box(person, region_name, min_vis)
            if not point_in_box(center, region):
                continue
            score = _closeness(center, region) * item.confidence
            candidates.append((score, item_idx, person_idx))

    candidates.sort(key=lambda c: c[0], reverse=True)

    used_items: set[int] = set()
    counts: dict[tuple[int, str], int] = defaultdict(int)
    assigned: dict[int, list[PPEItem]] = defaultdict(list)

    for _, item_idx, person_idx in candidates:
        if item_idx in used_items:
            continue
        item = ppe_items[item_idx]
        limit = policy.max_per_person.get(item.class_name, 1)
        if counts[(person_idx, item.class_name)] >= limit:
            continue
        used_items.add(item_idx)
        counts[(person_idx, item.class_name)] += 1
        assigned[person_idx].append(item)

    return [
        replace(person, ppe=[*person.ppe, *assigned.get(idx, [])])
        for idx, person in enumerate(persons)
    ]
