# src/core/models.py
"""Модели данных для проекта мониторинга СИЗ."""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class PPEItem:
    """Объект средства индивидуальной защиты (СИЗ)."""
    class_name: str
    box: List[int]  # [x1, y1, x2, y2]
    confidence: float

    def __str__(self) -> str:
        return f"PPE({self.class_name}, conf={self.confidence:.2f})"


@dataclass
class Person:
    """Объект человека с его атрибутами."""
    track_id: int
    box: List[int]  # [x1, y1, x2, y2]
    confidence: float
    keypoints: Optional[np.ndarray] = None
    assigned_ppe: List[PPEItem] = field(default_factory=list)


@dataclass
class Violation:
    """Запись о нарушении правил СИЗ."""
    person: Person
    missing_items: List[str]
    timestamp: float
    is_confirmed: bool = False
