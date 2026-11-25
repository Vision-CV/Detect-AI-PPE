"""Перевод точки на кадре в координаты пола (метры) через гомографию."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


class FloorProjector:
    def __init__(self, image_points: Sequence[Sequence[float]], world_points: Sequence[Sequence[float]]) -> None:
        src = np.asarray(image_points, dtype=np.float32)
        dst = np.asarray(world_points, dtype=np.float32)
        if src.shape != (4, 2) or dst.shape != (4, 2):
            raise ValueError("Нужно ровно 4 пары точек калибровки")
        self._h = cv2.getPerspectiveTransform(src, dst)

    def to_floor(self, x: float, y: float) -> tuple[float, float] | None:
        vx, vy, w = self._h @ np.array([x, y, 1.0])
        if abs(w) < 1e-9:
            return None
        return float(vx / w), float(vy / w)
