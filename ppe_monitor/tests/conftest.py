"""Общие фикстуры: синтетические видео и пути к реальным моделям/роликам."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ppe_monitor.config import PROJECT_ROOT

WEIGHTS = PROJECT_ROOT / "models_weights"
VIDEOS = PROJECT_ROOT / "test"
STEP = 8  # яркость кадра i = i * STEP — по ней узнаём номер кадра после сжатия


def write_video(path: Path, n_frames: int, fps: float = 10.0, size=(160, 120)) -> Path:
    """Видео, где кадр i — сплошная заливка яркостью i*STEP (MJPG — почти без потерь)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened(), "MJPG недоступен в этой сборке OpenCV"
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), (i * STEP) % 256, np.uint8))
    writer.release()
    return path


def frame_number(image: np.ndarray) -> int:
    return round(float(image.mean()) / STEP)


@pytest.fixture
def synthetic_video(tmp_path):
    return write_video(tmp_path / "synthetic.avi", 30)


def require_real_assets(*names: str) -> None:
    """Пропустить системный тест, если нет весов или тестовых роликов."""
    missing = [n for n in names if not (WEIGHTS / n).is_file() and not (VIDEOS / n).is_file()]
    if missing:
        pytest.skip(f"нет файлов: {missing}")
