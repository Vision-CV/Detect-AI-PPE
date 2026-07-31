# src/core/compliance.py
"""Модуль для проверки соответствия стандартам безопасности (СИЗ)."""

import time
from typing import List, Set, Optional

from src.core.models import Person, Violation
from config.settings import REQUIRED_PPE_CLASSES


def check_compliance(
    persons: List[Person],
    required_ppe: Optional[Set[str]] = None
) -> List[Violation]:
    """
    Проверяет наличие обязательных СИЗ у каждого человека.

    Args:
        persons: Список объектов Person с уже привязанными СИЗ.
        required_ppe: Множество требуемых классов (по умолчанию из settings).

    Returns:
        List[Violation]: Список зафиксированных нарушений.
    """
    if required_ppe is None:
        required_ppe = REQUIRED_PPE_CLASSES

    violations = []
    current_time = time.time()

    for person in persons:
        present_classes = {
            item.class_name.lower()
            for item in person.assigned_ppe
        }

        missing_classes = required_ppe - present_classes

        if missing_classes:
            violation = Violation(
                person=person,
                missing_items=list(missing_classes),
                timestamp=current_time
            )
            violations.append(violation)

    return violations
