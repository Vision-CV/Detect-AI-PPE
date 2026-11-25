"""Проверка: есть ли у человека все обязательные СИЗ."""

from __future__ import annotations

from ppe_monitor.domain.body import is_region_visible
from ppe_monitor.domain.models import Observation, ObsState, Person, PersonStatus, Rule, Status
from ppe_monitor.domain.policy import PPEPolicy


def evaluate(person: Person, policy: PPEPolicy) -> PersonStatus:
    """Различает три случая:

    - OK — все обязательные СИЗ найдены;
    - VIOLATION — чего-то нет, и соответствующая часть тела видна;
    - UNKNOWN — чего-то нет, но часть тела не видна (спиной, перекрыт,
      обрезан краем кадра). Это НЕ нарушение: «не вижу» ≠ «нет».
    """
    absent = policy.required - person.present_classes
    undetermined = frozenset(
        cls
        for cls in absent
        if not is_region_visible(
            person,
            policy.region_of(cls),
            policy.min_keypoint_visibility,
            policy.edge_margin_px,
        )
    )
    missing = absent - undetermined

    if missing:
        status = Status.VIOLATION
    elif undetermined:
        status = Status.UNKNOWN
    else:
        status = Status.OK
    return PersonStatus(person.track_id, status, missing, undetermined)


def ppe_observations(person: Person, status: PersonStatus, policy: PPEPolicy) -> list[Observation]:
    """По наблюдению на каждый обязательный класс СИЗ — у каждого свой эпизод."""
    result = []
    for cls in sorted(policy.required):
        if cls in status.missing:
            state = ObsState.VIOLATING
        elif cls in status.undetermined:
            state = ObsState.UNKNOWN
        else:
            state = ObsState.COMPLIANT
        result.append(Observation(person.track_id, Rule.PPE, cls, state, person.box, person.confidence))
    return result
