"""Правило «человек в запретной зоне»."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean

from ppe_monitor.domain.geometry import point_in_polygon, visible_points
from ppe_monitor.domain.models import FrameSize, Kp, Observation, ObsState, Person, Point, Rule, Zone


def foot_point(
    person: Person, frame_size: FrameSize, min_vis: float, edge_margin: float
) -> Point | None:
    """Точка, где человек стоит на полу (в пикселях), или None, если неизвестно.

    Приоритет — видимые лодыжки. Иначе низ бокса, но только если он не упирается
    в нижний край кадра: тогда ноги обрезаны и низ бокса — не пол.
    """
    ankles = visible_points(person.keypoints, Kp.ANKLES, min_vis)
    if ankles:
        return (mean(p[0] for p in ankles), max(p[1] for p in ankles))

    x1, _, x2, y2 = person.box
    if y2 >= frame_size[1] - edge_margin:
        return None
    return ((x1 + x2) / 2, y2)


def zone_observations(
    persons: Sequence[Person],
    zones: Sequence[Zone],
    frame_size: FrameSize,
    min_vis: float,
    edge_margin: float,
) -> list[Observation]:
    """Для каждого человека и каждой зоны: внутри / снаружи / неизвестно (ноги не видны)."""
    if not zones:
        return []
    width, height = frame_size
    result = []
    for person in persons:
        foot = foot_point(person, frame_size, min_vis, edge_margin)
        normalized = None if foot is None else (foot[0] / width, foot[1] / height)
        for zone in zones:
            if normalized is None:
                state = ObsState.UNKNOWN
            elif point_in_polygon(normalized, zone.polygon):
                state = ObsState.VIOLATING
            else:
                state = ObsState.COMPLIANT
            result.append(Observation(person.track_id, Rule.ZONE, zone.name, state, person.box, person.confidence))
    return result
