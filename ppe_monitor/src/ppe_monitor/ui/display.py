"""Окно предпросмотра OpenCV."""

from __future__ import annotations

import cv2
import numpy as np


class Window:
    def __init__(self, name: str) -> None:
        self._name = name
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    def show(self, image: np.ndarray) -> bool:
        """Показывает кадр. Возвращает False, если пользователь закрыл окно или нажал q/Esc."""
        cv2.imshow(self._name, image)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        return cv2.getWindowProperty(self._name, cv2.WND_PROP_VISIBLE) >= 1

    def close(self) -> None:
        cv2.destroyWindow(self._name)
