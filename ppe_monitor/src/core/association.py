# src/core/association.py
"""Модуль для ассоциации (привязки) объектов СИЗ к людям."""

import numpy as np
from typing import List

from src.core.models import Person, PPEItem


def calculate_iou(box_a: List[int], box_b: List[int]) -> float:
    """
    Вычисляет IoU (Intersection over Union) для двух боксов.

    Args:
        box_a: Первый бокс [x1, y1, x2, y2].
        box_b: Второй бокс [x1, y1, x2, y2].

    Returns:
        float: Значение IoU от 0.0 до 1.0.
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    return inter_area / (area_a + area_b - inter_area)


def _is_point_in_box(point: List[float], box: List[int]) -> bool:
    """
    Проверяет, находится ли точка внутри прямоугольника

    Args:
        point: Координаты точки [x, y]
        box: Координаты прямоугольника [x1, y1, x2, y2]

    Returns:
        bool: True, если точка внутри
    """

    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def associate_ppe(
        persons: List[Person], ppe_items: List[PPEItem]) -> List[Person]:
    """
    Привязывает обнаруженные СИЗ к соответсвующим людям.
    Модифицирует объекты Person на месте (добавляет в assigned_ppe).

    Алгоритм приоритетов: 
    1. Центр СИЗ внутри бокса человека.
    2. Расстояние до ключевых точек (голова/торс).
    3. IoU (перекрытие боксов).

    Args:
        persons: Список людей для проверки соответсвий.
        ppe_items:  Список обнаруженных объектов СИЗ.

    Retuens:
        List[Person]: Обновленный список людей.
    """
    sorted_ppe = sorted(
        ppe_items,
        key=lambda item: item.confidence,
        reverse=True
        )

    for ppe in sorted_ppe:
        ppe_center = [
            (ppe.box[0] + ppe.box[2]) / 2,
            (ppe.box[1] + ppe.box[3]) / 2
        ]

        best_person_idx = -1
        best_score = -1.0

        for idx, person in enumerate(persons):
            person_width = person.box[2] - person.box[0]
            person_height = person.box[3] - person.box[1]
            # Проверка на удаленность человека
            is_small_person = person_width < 100

            if ppe_center[1] > person.box[3] + 50:
                continue

            if ppe.class_name == "vest" and ppe.box[2] - ppe.box[0] > person_width * 1.5:
                continue
            if ppe.class_name == "helmet" and ppe.box[2] - ppe.box[0] > person_width * 1.2:
                continue

            score = 0.0

            if _is_point_in_box(ppe_center, person.box):
                score += 200.0

            target_indices = []
            if ppe.class_name == "helmet":
                target_indices = [0, 1, 2, 3, 4]
            elif ppe.class_name == "vest":
                target_indices = [5, 6, 11, 12]

            valid_points_count = 0
            total_dist = 0.0

            if person.keypoints is not None:
                for pt_idx in target_indices:
                    visibility_thresh = 0.2 if is_small_person else 0.3
                    if person.keypoints[pt_idx][2] < visibility_thresh:
                        continue

                    valid_points_count += 1
                    kp_x, kp_y = person.keypoints[pt_idx][0], person.keypoints[pt_idx][1]
                    dist = np.sqrt((kp_x - ppe_center[0])**2 + (kp_y - ppe_center[1])**2)

                    max_dist = person_height * 0.6
                    if dist < max_dist:
                        total_dist += dist

            if valid_points_count > 0:
                avg_dist = total_dist / valid_points_count
                score += 50.0 / (1.0 + avg_dist)

            if score < 50.0:
                iou = calculate_iou(ppe.box, person.box)
                req_iou = 0.05 if is_small_person else 0.3
                if iou > req_iou:
                    score += iou * 100

            if score > best_score:
                best_score = score
                best_person_idx = idx

        if best_person_idx != -1 and best_score > 30.0:
            persons[best_person_idx].assigned_ppe.append(ppe)

    return persons
