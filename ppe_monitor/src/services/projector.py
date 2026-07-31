# src/services/projector.py
"""Сервис для проекции координат с камеры на 2D карту (пол)."""

import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ProjectorService:
    """Сервис калибровки и перевода пикселей в метры."""

    def __init__(self, img_pts: List[List[float]], world_pts: List[List[float]]) -> None:
        """
        Инициализирует матрицу гомографии на основе калибровочных точек.

        Args:
            img_pts: 4 точки на изображении [[x,y], ...].
            world_pts: 4 точки в реальном мире в метрах [[x,z], ...].
        """
        if len(img_pts) != 4 or len(world_pts) != 4:
            raise ValueError("Требуется ровно 4 калибровочные точки для проекции.")

        src = np.array(img_pts, dtype=np.float32)
        dst = np.array(world_pts, dtype=np.float32)

        self.H, _ = cv2.findHomography(src, dst)
        logger.info("✅ Floor Projector initialized.")

    def pixel_to_meter(self, px: float, py: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Переводит пиксельные координаты в метры.

        Args:
            px: X в пикселях.
            py: Y в пикселях.

        Returns:
            Tuple: (x_meters, z_meters) или (None, None).
        """
        point = np.array([px, py, 1], dtype=np.float32)
        transformed = np.dot(self.H, point)

        if transformed[2] != 0:
            transformed /= transformed[2]
            return float(transformed[0]), float(transformed[1])
        return None, None
