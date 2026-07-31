# src/ui/renderer.py
"""Модуль для отрисовки результатов на кадре (OpenCV)."""

import logging
from typing import List, Optional, Set

import cv2
import numpy as np

from src.core.models import Person, Violation
from src.services.projector import ProjectorService

logger = logging.getLogger(__name__)

COLOR_OK = (0, 255, 0)
COLOR_PENDING = (0, 165, 255)
COLOR_CONFIRMED = (0, 0, 255)
COLOR_PPE = (255, 0, 0)
COLOR_TEXT_BG = (0, 0, 0)


def draw_frame(
    frame: np.ndarray,
    persons: List[Person],
    violations: List[Violation],
    confirmed_ids: Set[int],
    projector: Optional[ProjectorService] = None,
    fps: float = 0.0
) -> np.ndarray:
    """
    Отрисовывает детекции, треки и информацию на кадре.

    Args:
        frame: Исходный кадр (BGR).
        persons: Список людей с привязанными СИЗ.
        violations: Список текущих нарушений.
        confirmed_ids: ID людей, чьи нарушения уже подтверждены (для красной рамки).
        projector: Опциональный сервис проекции координат.
        fps: Текущий FPS для отображения.

    Returns:
        np.ndarray: Кадр с отрисованной информацией.
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    violating_ids = {v.person.track_id for v in violations}

    for person in persons:
        x1, y1, x2, y2 = person.box
        tid = person.track_id

        # Выбираем цвет рамки
        if tid in confirmed_ids:
            color = COLOR_CONFIRMED
        elif tid in violating_ids:
            color = COLOR_PENDING
        else:
            color = COLOR_OK

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{tid} {person.confidence:.2f}"

        if projector:
            foot_x = (x1 + x2) / 2
            foot_y = y2
            mx, mz = projector.pixel_to_meter(foot_x, foot_y)
            if mx is not None:
                label += f" [{mx:.1f}m, {mz:.1f}m]"
                # Точка на полу
                cv2.circle(vis, (int(foot_x), int(foot_y)), 3, (255, 255, 0), -1)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 5), (x1 + tw, y1), COLOR_TEXT_BG, -1)
        cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if person.keypoints is not None:
            for kp in person.keypoints:
                if kp[2] > 0.5:  # Рисуем только видимые точки
                    cv2.circle(vis, (int(kp[0]), int(kp[1])), 2, (255, 255, 0), -1)

        for ppe in person.assigned_ppe:
            px1, py1, px2, py2 = ppe.box
            cv2.rectangle(vis, (px1, py1), (px2, py2), COLOR_PPE, 1)
            cv2.putText(
                vis, ppe.class_name, (px1, py2 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_PPE, 1
            )

    cv2.putText(
        vis, f"FPS: {fps:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    return vis
