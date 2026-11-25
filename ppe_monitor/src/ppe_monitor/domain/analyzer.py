"""Анализ одного кадра: привязка СИЗ + все правила → наблюдения.

Новое правило (например, «человек лежит») добавляется сюда отдельной
функцией, возвращающей Observation'ы. Остальной конвейер менять не нужно.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ppe_monitor.domain.association import associate
from ppe_monitor.domain.compliance import evaluate, ppe_observations
from ppe_monitor.domain.models import FrameSize, Observation, ObsState, Person, PersonStatus, PPEItem, Zone
from ppe_monitor.domain.policy import PPEPolicy
from ppe_monitor.domain.zones import zone_observations


@dataclass(frozen=True)
class FrameAnalysis:
    persons: list[Person]  # с привязанными СИЗ
    statuses: dict[int, PersonStatus]  # track_id -> статус по СИЗ
    observations: list[Observation]  # наблюдения всех правил

    def violating(self) -> list[Observation]:
        return [o for o in self.observations if o.state is ObsState.VIOLATING]


class Analyzer:
    def __init__(self, policy: PPEPolicy, zones: Sequence[Zone] = ()) -> None:
        self.policy = policy
        self.zones = tuple(zones)

    def analyze(
        self,
        persons: Sequence[Person],
        ppe_items: Sequence[PPEItem],
        frame_size: FrameSize,
    ) -> FrameAnalysis:
        persons = associate(persons, ppe_items, self.policy)

        statuses: dict[int, PersonStatus] = {}
        observations: list[Observation] = []
        for person in persons:
            status = evaluate(person, self.policy)
            statuses[person.track_id] = status
            observations.extend(ppe_observations(person, status, self.policy))

        observations.extend(
            zone_observations(
                persons,
                self.zones,
                frame_size,
                self.policy.min_keypoint_visibility,
                self.policy.edge_margin_px,
            )
        )
        return FrameAnalysis(persons, statuses, observations)
