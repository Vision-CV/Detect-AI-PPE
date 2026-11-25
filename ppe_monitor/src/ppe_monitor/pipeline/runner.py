"""Главный цикл: только соединяет стадии. Все решения принимаются в domain/."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from ppe_monitor.domain.analyzer import Analyzer, FrameAnalysis
from ppe_monitor.domain.episode_tracker import EpisodeTracker
from ppe_monitor.domain.models import EpisodeEvent, EpisodeKind
from ppe_monitor.pipeline.detector import Detector
from ppe_monitor.pipeline.evidence import EvidenceCollector
from ppe_monitor.pipeline.source import Frame
from ppe_monitor.ui.renderer import draw

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    frames: int = 0
    opened: int = 0
    closed: int = 0
    seconds: float = 0.0

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds else 0.0


class Pipeline:
    def __init__(
        self,
        detector: Detector,
        analyzer: Analyzer,
        tracker: EpisodeTracker,
        evidence: EvidenceCollector,
        *,
        show: Callable[[np.ndarray], bool] | None = None,
    ) -> None:
        self.detector = detector
        self.analyzer = analyzer
        self.tracker = tracker
        self.evidence = evidence
        self.show = show
        self._fps: float | None = None
        self.stats = RunStats()

    def run(
        self,
        frames: Iterable[Frame],
        stop: threading.Event | None = None,
        max_frames: int | None = None,
    ) -> RunStats:
        started = time.perf_counter()
        try:
            for frame in frames:
                if stop is not None and stop.is_set():
                    logger.info("Остановка по сигналу")
                    break
                t0 = time.perf_counter()
                analysis, _ = self.process(frame)
                self._update_fps(time.perf_counter() - t0)

                if self.show is not None and not self.show(self._render(frame, analysis, with_fps=True)):
                    logger.info("Окно закрыто пользователем")
                    break
                if max_frames is not None and self.stats.frames >= max_frames:
                    break
        finally:
            # Конец видео / остановка: открытые эпизоды закрываются с причиной stream_ended.
            self._count(closing := self.tracker.flush())
            self.evidence.on_flush(closing)
            self.stats.seconds = time.perf_counter() - started
        return self.stats

    def process(self, frame: Frame) -> tuple[FrameAnalysis, list[EpisodeEvent]]:
        """Обрабатывает один кадр. Отдельный метод — чтобы его можно было тестировать."""
        persons, ppe_items = self.detector.detect(frame.image)
        analysis = self.analyzer.analyze(persons, ppe_items, frame.size)
        events = self.tracker.update(frame.timestamp, analysis.observations)
        self.evidence.on_frame(
            frame,
            analysis,
            events,
            render=lambda: self._render(frame, analysis, with_fps=False),
            open_episodes=self.tracker.open_episodes(),
        )
        self.stats.frames += 1
        self._count(events)
        return analysis, events

    def _count(self, events: list[EpisodeEvent]) -> None:
        for e in events:
            if e.kind is EpisodeKind.OPENED:
                self.stats.opened += 1
                logger.warning("НАРУШЕНИЕ: %s, ID %d", e.violation_type, e.track_id)
            elif e.kind is EpisodeKind.CLOSED:
                self.stats.closed += 1
                logger.info(
                    "Нарушение завершено: %s, ID %s, %.1f с (%s)",
                    e.violation_type, "/".join(map(str, e.track_ids)), e.duration_sec or 0, e.close_reason.value,
                )

    def _render(self, frame: Frame, analysis: FrameAnalysis, with_fps: bool) -> np.ndarray:
        return draw(
            frame.image,
            analysis,
            confirmed_track_ids=self.tracker.open_track_ids(),
            zones=self.analyzer.zones,
            fps=self._fps if with_fps else None,
            min_keypoint_visibility=self.analyzer.policy.min_keypoint_visibility,
        )

    def _update_fps(self, dt: float) -> None:
        if dt <= 0:
            return
        current = 1.0 / dt
        self._fps = current if self._fps is None else 0.9 * self._fps + 0.1 * current
