"""Параметры правил проверки СИЗ (что требуется и где оно должно быть)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


class Region:
    """Части тела, к которым привязываются СИЗ."""

    HEAD = "head"
    TORSO = "torso"
    BODY = "body"  # весь человек (перчатки, ботинки и т.п.)

    ALL = (HEAD, TORSO, BODY)


@dataclass(frozen=True)
class PPEPolicy:
    required: frozenset[str] = frozenset({"helmet", "vest"})
    # Класс СИЗ -> часть тела. Не указанные классы считаются Region.BODY.
    regions: Mapping[str, str] = field(
        default_factory=lambda: {"helmet": Region.HEAD, "vest": Region.TORSO}
    )
    # Класс -> максимальная ширина СИЗ относительно ширины человека.
    max_width_ratio: Mapping[str, float] = field(
        default_factory=lambda: {"helmet": 1.2, "vest": 1.6}
    )
    # Класс -> сколько штук может быть у одного человека (по умолчанию 1).
    max_per_person: Mapping[str, int] = field(
        default_factory=lambda: {"gloves": 2, "boots": 2}
    )
    min_keypoint_visibility: float = 0.3
    # Если бокс человека ближе этого расстояния к краю кадра — часть тела обрезана.
    edge_margin_px: float = 4.0

    def region_of(self, class_name: str) -> str:
        return self.regions.get(class_name, Region.BODY)
