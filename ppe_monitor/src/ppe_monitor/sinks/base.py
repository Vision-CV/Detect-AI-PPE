"""Пакет инцидента — событие эпизода + контекст + файлы-доказательства.

Пакет самодостаточен: по нему без доступа к камере можно понять, кто,
где, когда и как долго нарушал, и проверить целостность файлов (SHA-256).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from ppe_monitor.domain.models import EpisodeEvent

logger = logging.getLogger(__name__)

SCHEMA = "ppe-incident/1"


@dataclass(frozen=True)
class EvidenceFile:
    name: str  # имя файла в пакете: "frame_start.jpg"
    kind: str  # frame_start | frame_best | frame_end | frame_current | raw_start | clip
    data: bytes
    content_type: str
    local_only: bool = False  # не отправлять (например, кадр без размытия лиц)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "content_type": self.content_type,
            "size": len(self.data),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class IncidentPackage:
    event: EpisodeEvent
    files: tuple[EvidenceFile, ...]
    source: str
    frame_index: int
    started_at: datetime
    event_at: datetime
    ended_at: datetime | None = None
    video_offsets_ms: dict[str, int | None] | None = None  # только для видеофайлов
    present_classes: tuple[str, ...] = ()
    position_m: tuple[float, float] | None = None
    model: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return self.event.kind.value

    @property
    def is_alert(self) -> bool:
        """Срочное сообщение (открытие, напоминание) — отправляется раньше тяжёлых пакетов."""
        return self.status != "closed"

    def upload_files(self) -> list[EvidenceFile]:
        return [f for f in self.files if not f.local_only]

    def manifest(self, *, include_local: bool = False) -> dict[str, Any]:
        e = self.event
        files = self.files if include_local else self.upload_files()
        return {
            "schema": SCHEMA,
            "episode_id": e.episode_id,
            "status": self.status,
            "camera_id": e.camera_id,
            "source": self.source,
            "rule": e.rule,
            "subject": e.subject,
            "violation_type": e.violation_type,
            "track_id": e.track_id,
            "track_ids": list(e.track_ids),
            "started_at": _iso(self.started_at),
            "event_at": _iso(self.event_at),
            "ended_at": _iso(self.ended_at),
            "duration_sec": None if e.duration_sec is None else round(e.duration_sec, 2),
            "violating_sec": round(e.violating_sec, 2),
            "close_reason": e.close_reason.value if e.close_reason else None,
            "confidence": round(e.confidence, 3),
            "person_box": [round(v) for v in e.box],
            "present_classes": list(self.present_classes),
            "position_2d_meters": (
                {"x": round(self.position_m[0], 2), "z": round(self.position_m[1], 2)}
                if self.position_m
                else None
            ),
            "frame_index": self.frame_index,
            "video_offsets_ms": self.video_offsets_ms,
            "model": self.model,
            "evidence": [f.describe() for f in files],
        }


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat(timespec="milliseconds")


class PackageSink(Protocol):
    """Получатель пакетов. `publish` не должен блокировать конвейер надолго."""

    def publish(self, package: IncidentPackage) -> None: ...

    def close(self) -> None: ...


class Broadcast:
    """Раздаёт пакет всем получателям. Сбой одного не мешает остальным."""

    def __init__(self, sinks: Sequence[PackageSink]) -> None:
        self.sinks = list(sinks)

    def publish(self, package: IncidentPackage) -> None:
        for sink in self.sinks:
            try:
                sink.publish(package)
            except Exception:
                logger.exception("Ошибка публикации в %s", type(sink).__name__)

    def close(self) -> None:
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:
                logger.exception("Ошибка закрытия %s", type(sink).__name__)
