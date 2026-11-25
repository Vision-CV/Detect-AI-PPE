"""Сбор доказательств и сборка пакетов инцидентов.

Жизненный цикл пакетов одного эпизода:

    OPENED   → сразу: кадр подтверждения (с разметкой и исходный). Срочно.
    UPDATED  → сразу: текущий кадр (если включены напоминания).
    CLOSED   → кадры начала / «лучший» / конца + клип ±N с вокруг подтверждения.

Тревога (OPENED) никогда не ждёт клипа: клип дописывается и кодируется в
фоновом потоке и уходит в пакете CLOSED.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import cv2
import numpy as np

from ppe_monitor.config import EvidenceConfig
from ppe_monitor.domain.analyzer import FrameAnalysis
from ppe_monitor.domain.body import head_region
from ppe_monitor.domain.geometry import box_height, box_width
from ppe_monitor.domain.models import Box, CloseReason, EpisodeEvent, EpisodeKind, Person
from ppe_monitor.pipeline.source import Frame
from ppe_monitor.sinks.base import EvidenceFile, IncidentPackage
from ppe_monitor.ui.renderer import encode_jpeg

logger = logging.getLogger(__name__)

JPEG = "image/jpeg"
MP4 = "video/mp4"
CLIP_BOX_COLOR = (0, 0, 255)


@dataclass
class _ClipFrame:
    ts: float
    jpeg: bytes  # уменьшенный кадр (без разметки)
    boxes: dict[int, Box]  # track_id -> бокс в координатах уменьшенного кадра
    heads: list[Box]  # области голов для размытия (если включено)


@dataclass
class _Episode:
    first: EpisodeEvent
    started_at: datetime
    start_jpeg: bytes
    raw_jpeg: bytes
    track_ids: set[int]
    best_jpeg: bytes | None = None
    best_score: float = 0.0
    clip_frames: list[_ClipFrame] = field(default_factory=list)
    clip_until: float | None = None  # пока ts <= clip_until — дописываем кадры
    clip_future: Any = None  # Future[bytes | None]


class EvidenceCollector:
    def __init__(
        self,
        config: EvidenceConfig,
        publish: Callable[[IncidentPackage], None],
        *,
        source_name: str = "",
        model_info: dict[str, Any] | None = None,
        min_keypoint_visibility: float = 0.3,
        locate: Callable[[Person, tuple[int, int]], tuple[float, float] | None] | None = None,
    ) -> None:
        self._cfg = config
        self._publish = publish
        self._source = source_name
        self._model = model_info or {}
        self._min_vis = min_keypoint_visibility
        self._locate = locate
        self._buffer: deque[_ClipFrame] = deque()
        self._episodes: dict[str, _Episode] = {}
        # Один рабочий поток: клипы и пакеты CLOSED выходят строго по порядку.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Evidence")
        self._last_frame: Frame | None = None

    # ------------------------------------------------------------ публичное

    def on_frame(
        self,
        frame: Frame,
        analysis: FrameAnalysis,
        events: Iterable[EpisodeEvent],
        render: Callable[[], np.ndarray],
        open_episodes: dict[str, tuple[int, str, str]],
    ) -> None:
        """Вызывается на каждом обработанном кадре.

        render — ленивая отрисовка кадра с разметкой (рисуем только если нужен снимок).
        open_episodes — episode_id -> (track_id, правило, предмет) из трекера.
        """
        self._last_frame = frame
        annotated: list[np.ndarray] = []

        def annotated_frame() -> np.ndarray:
            if not annotated:
                annotated.append(self._blur(render(), analysis.persons))
            return annotated[0]

        clip_frame = self._buffer_frame(frame, analysis) if self._cfg.clip_enabled else None

        for event in events:
            if event.kind is EpisodeKind.OPENED:
                self._on_opened(event, frame, analysis, annotated_frame)
            elif event.kind is EpisodeKind.UPDATED:
                self._on_updated(event, frame, analysis, annotated_frame)
            else:
                self._on_closed(event, frame, analysis, annotated_frame)

        self._track_best(frame, analysis, open_episodes, annotated_frame)
        if clip_frame is not None:
            self._extend_clips(clip_frame)

    def on_flush(self, events: Iterable[EpisodeEvent]) -> None:
        """События закрытия при остановке (кадра уже нет — без кадра конца)."""
        for event in events:
            self._on_closed(event, self._last_frame, None, None)

    def close(self) -> None:
        self._worker.shutdown(wait=True)

    # ------------------------------------------------------------ события

    def _on_opened(self, event, frame, analysis, annotated_frame) -> None:
        start_jpeg = encode_jpeg(annotated_frame(), self._cfg.jpeg_quality)
        raw_jpeg = encode_jpeg(frame.image, self._cfg.jpeg_quality)
        ep = _Episode(
            first=event,
            started_at=self._wall(frame, event.started_ts),
            start_jpeg=start_jpeg,
            raw_jpeg=raw_jpeg,
            track_ids=set(event.track_ids),
        )
        if self._cfg.clip_enabled:
            ep.clip_frames = [f for f in self._buffer if f.ts >= event.event_ts - self._cfg.clip_pre_sec]
            ep.clip_until = event.event_ts + self._cfg.clip_post_sec
        self._episodes[event.episode_id] = ep

        files = [EvidenceFile("frame_start.jpg", "frame_start", start_jpeg, JPEG), self._raw_file(raw_jpeg)]
        self._publish(self._package(event, frame, analysis, files, ep))

    def _on_updated(self, event, frame, analysis, annotated_frame) -> None:
        ep = self._episodes.get(event.episode_id)
        current = encode_jpeg(annotated_frame(), self._cfg.jpeg_quality)
        files = [EvidenceFile("frame_current.jpg", "frame_current", current, JPEG)]
        self._publish(self._package(event, frame, analysis, files, ep))

    def _on_closed(self, event, frame, analysis, annotated_frame) -> None:
        ep = self._episodes.pop(event.episode_id, None)
        files: list[EvidenceFile] = []
        if ep is not None:
            files.append(EvidenceFile("frame_start.jpg", "frame_start", ep.start_jpeg, JPEG))
            files.append(self._raw_file(ep.raw_jpeg))
            if ep.best_jpeg is not None:
                files.append(EvidenceFile("frame_best.jpg", "frame_best", ep.best_jpeg, JPEG))
        if annotated_frame is not None and event.close_reason is CloseReason.RESOLVED:
            end = encode_jpeg(annotated_frame(), self._cfg.jpeg_quality)
            files.append(EvidenceFile("frame_end.jpg", "frame_end", end, JPEG))

        if ep is not None and ep.clip_until is not None:
            self._finish_clip(event.episode_id, ep)  # эпизод кончился раньше, чем запись клипа
        clip_future = ep.clip_future if ep is not None else None

        def publish_closed() -> None:
            all_files = list(files)
            if clip_future is not None:
                clip = clip_future.result()
                if clip:
                    all_files.append(EvidenceFile("clip.mp4", "clip", clip, MP4))
            self._publish(self._package(event, frame, analysis, all_files, ep))

        # Через тот же поток, что и кодирование клипа: пакет уйдёт после клипа.
        self._worker.submit(self._safe, publish_closed)

    # ------------------------------------------------------------ лучший кадр

    def _track_best(self, frame, analysis, open_episodes, annotated_frame) -> None:
        if not open_episodes:
            return
        persons = {p.track_id: p for p in analysis.persons}
        violating = {o.key for o in analysis.violating()}
        for episode_id, key in open_episodes.items():
            ep = self._episodes.get(episode_id)
            person = persons.get(key[0])
            if ep is None or person is None or key not in violating:
                continue
            ep.track_ids.add(key[0])
            score = box_width(person.box) * box_height(person.box) * person.confidence
            if score > ep.best_score * 1.15:  # заметно лучше — перекодируем
                ep.best_score = score
                ep.best_jpeg = encode_jpeg(annotated_frame(), self._cfg.jpeg_quality)

    # ------------------------------------------------------------ клип

    def _buffer_frame(self, frame: Frame, analysis: FrameAnalysis) -> _ClipFrame:
        h, w = frame.image.shape[:2]
        scale = min(1.0, self._cfg.clip_width / w)
        small = cv2.resize(frame.image, (round(w * scale), round(h * scale))) if scale < 1 else frame.image
        boxes = {p.track_id: tuple(v * scale for v in p.box) for p in analysis.persons}
        heads = (
            [tuple(v * scale for v in head_region(p, self._min_vis)) for p in analysis.persons]
            if self._cfg.blur_faces
            else []
        )
        item = _ClipFrame(frame.timestamp, encode_jpeg(small, 85), boxes, heads)
        self._buffer.append(item)
        while self._buffer and self._buffer[0].ts < frame.timestamp - self._cfg.clip_pre_sec:
            self._buffer.popleft()
        return item

    def _extend_clips(self, clip_frame: _ClipFrame) -> None:
        for episode_id, ep in list(self._episodes.items()):
            if ep.clip_until is None:
                continue
            if clip_frame.ts <= ep.clip_until:
                if not ep.clip_frames or ep.clip_frames[-1] is not clip_frame:
                    ep.clip_frames.append(clip_frame)
            else:
                self._finish_clip(episode_id, ep)

    def _finish_clip(self, episode_id: str, ep: _Episode) -> None:
        frames, ep.clip_frames, ep.clip_until = ep.clip_frames, [], None
        label = f"{ep.first.violation_type} | episode {episode_id[:8]}"
        ep.clip_future = self._worker.submit(
            self._safe, write_clip, frames, set(ep.track_ids), label, self._cfg.clip_codec, self._cfg.blur_faces
        )

    # ------------------------------------------------------------ сборка

    def _package(self, event, frame, analysis, files, ep) -> IncidentPackage:
        frame = frame or self._last_frame
        person = None
        if analysis is not None:
            person = next((p for p in analysis.persons if p.track_id == event.track_id), None)
        position = None
        if person is not None and self._locate is not None and frame is not None:
            position = self._locate(person, frame.size)

        is_file = frame is not None and frame.video_pos_ms is not None
        offsets = None
        if is_file:
            offsets = {
                "started": round(event.started_ts * 1000),
                "event": round(event.event_ts * 1000),
                "ended": None if event.ended_ts is None else round(event.ended_ts * 1000),
            }
        return IncidentPackage(
            event=event,
            files=tuple(files),
            source=self._source,
            frame_index=frame.index if frame is not None else -1,
            started_at=ep.started_at if ep is not None else self._wall(frame, event.started_ts),
            event_at=self._wall(frame, event.event_ts),
            ended_at=None if event.ended_ts is None else self._wall(frame, event.ended_ts),
            video_offsets_ms=offsets,
            present_classes=tuple(sorted(person.present_classes)) if person else (),
            position_m=position,
            model=self._model,
        )

    def _raw_file(self, raw_jpeg: bytes) -> EvidenceFile:
        # С размытием лиц исходный кадр остаётся только локально.
        return EvidenceFile("raw_start.jpg", "raw_start", raw_jpeg, JPEG, local_only=self._cfg.blur_faces)

    def _blur(self, image: np.ndarray, persons: Sequence[Person]) -> np.ndarray:
        if not self._cfg.blur_faces:
            return image
        return blur_regions(image, [head_region(p, self._min_vis) for p in persons])

    @staticmethod
    def _wall(frame: Frame | None, ts: float) -> datetime:
        """Время кадра ts в «настенных» часах, по опорному кадру."""
        if frame is None:
            return datetime.now()
        return frame.captured_at - timedelta(seconds=frame.timestamp - ts)

    @staticmethod
    def _safe(fn, *args):
        try:
            return fn(*args)
        except Exception:
            logger.exception("Ошибка при сборке доказательств")
            return None


# ---------------------------------------------------------------- утилиты


def blur_regions(image: np.ndarray, boxes: Sequence[Box]) -> np.ndarray:
    """Сильное размытие прямоугольников (головы) — для отправляемых кадров."""
    out = image.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = out[y1:y2, x1:x2]
        k = max(3, (min(x2 - x1, y2 - y1) // 3) | 1)
        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


def write_clip(
    frames: Sequence[_ClipFrame], track_ids: set[int], label: str, codec: str, blur: bool
) -> bytes | None:
    """Собирает mp4 из кадров буфера, обводит нарушителя. None — если кадров нет."""
    if not frames:
        return None
    first = cv2.imdecode(np.frombuffer(frames[0].jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    span = frames[-1].ts - frames[0].ts
    fps = min(30.0, max(1.0, (len(frames) - 1) / span)) if span > 0 else 5.0

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if not writer.isOpened():
            logger.error("Кодек %s недоступен — клип не записан", codec)
            return None
        for item in frames:
            img = cv2.imdecode(np.frombuffer(item.jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            if blur and item.heads:
                img = blur_regions(img, item.heads)
            for tid, (x1, y1, x2, y2) in item.boxes.items():
                if tid in track_ids:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), CLIP_BOX_COLOR, 2)
            cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLIP_BOX_COLOR, 2, cv2.LINE_AA)
            writer.write(img)
        writer.release()
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)
