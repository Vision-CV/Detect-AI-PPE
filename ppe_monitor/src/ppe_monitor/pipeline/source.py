"""Источники кадров: видеофайл и «живой» поток (камера, RTSP/HTTP).

Они ведут себя по-разному, поэтому это два класса:

- Файл читаем по порядку; пропущенные кадры только `grab()` (без декодирования);
  время кадра — позиция в видео. Обработка может идти быстрее или медленнее
  реального времени, логика нарушений от этого не меняется.
- Поток читаем в фоновом потоке и всегда отдаём САМЫЙ СВЕЖИЙ кадр — иначе при
  медленном инференсе буфер копится и задержка растёт. При обрыве —
  переподключение. Время кадра — монотонные часы.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ppe_monitor.config import SourceConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    index: int
    image: np.ndarray
    timestamp: float  # секунды; используется логикой нарушений
    captured_at: datetime  # «настенное» время для журналов
    video_pos_ms: float | None = None  # только для файлов

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return (w, h)


class SourceError(RuntimeError):
    pass


def _resize(image: np.ndarray, width: int | None) -> np.ndarray:
    if not width:
        return image
    h, w = image.shape[:2]
    if w == width:
        return image
    return cv2.resize(image, (width, round(h * width / w)))


def _is_live(uri: str) -> bool:
    return uri.isdigit() or uri.split("://")[0].lower() in {"rtsp", "rtmp", "http", "https", "udp", "tcp"}


def _open_capture(uri: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(int(uri) if uri.isdigit() else uri)


class FileSource:
    def __init__(self, path: Path, every_n: int, resize_width: int | None) -> None:
        if not path.is_file():
            raise SourceError(f"Видеофайл не найден: {path}")
        self.name = str(path)
        self._cap = _open_capture(str(path))
        if not self._cap.isOpened():
            raise SourceError(f"Не удалось открыть видео: {path}")
        self._every_n = every_n
        self._resize_width = resize_width
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0

    def __iter__(self) -> Iterator[Frame]:
        index = -1
        while True:
            index += 1
            if index % self._every_n:
                if not self._cap.grab():
                    break
                continue
            ok, image = self._cap.read()
            if not ok:
                break
            pos_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms <= 0 and self._fps > 0:
                pos_ms = index * 1000.0 / self._fps
            yield Frame(
                index=index,
                image=_resize(image, self._resize_width),
                timestamp=pos_ms / 1000.0,
                captured_at=datetime.now(),
                video_pos_ms=pos_ms,
            )
        logger.info("Видео закончилось: %s", self.name)

    def close(self) -> None:
        self._cap.release()


class LiveSource:
    def __init__(self, uri: str, resize_width: int | None, reconnect_attempts: int, reconnect_delay: float) -> None:
        self.name = uri
        self._uri = uri
        self._resize_width = resize_width
        self._attempts = reconnect_attempts
        self._delay = reconnect_delay

        self._cap = _open_capture(uri)
        if not self._cap.isOpened():
            raise SourceError(f"Не удалось открыть поток: {uri}")

        self._cond = threading.Condition()
        self._latest: np.ndarray | None = None
        self._latest_seq = 0
        self._finished = False
        self._closing = threading.Event()
        self._thread = threading.Thread(target=self._reader, name="VideoReader", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        failures = 0
        while not self._closing.is_set():
            ok, image = self._cap.read()
            if ok:
                failures = 0
                with self._cond:
                    self._latest = image
                    self._latest_seq += 1
                    self._cond.notify_all()
                continue

            failures += 1
            if self._attempts and failures > self._attempts:
                logger.error("Поток %s недоступен после %d попыток — остановка", self._uri, self._attempts)
                break
            logger.warning("Поток %s оборвался, переподключение %d…", self._uri, failures)
            self._cap.release()
            if self._closing.wait(self._delay):
                break
            self._cap = _open_capture(self._uri)

        with self._cond:
            self._finished = True
            self._cond.notify_all()

    def __iter__(self) -> Iterator[Frame]:
        start = time.monotonic()
        seen_seq = 0
        index = 0
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._latest_seq != seen_seq or self._finished, timeout=1.0)
                if self._latest_seq == seen_seq:
                    if self._finished:
                        return
                    continue
                image, seen_seq = self._latest, self._latest_seq
            yield Frame(
                index=index,
                image=_resize(image, self._resize_width),
                timestamp=time.monotonic() - start,
                captured_at=datetime.now(),
            )
            index += 1

    def close(self) -> None:
        self._closing.set()
        self._thread.join(timeout=5)
        self._cap.release()


def open_source(config: SourceConfig, every_n: int) -> FileSource | LiveSource:
    if _is_live(config.uri):
        return LiveSource(config.uri, config.resize_width, config.reconnect_attempts, config.reconnect_delay_sec)
    return FileSource(Path(config.uri), every_n, config.resize_width)
