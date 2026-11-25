"""Модели данных, которыми обмениваются стадии конвейера."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# (x1, y1, x2, y2) в пикселях кадра.
Box = tuple[float, float, float, float]
Point = tuple[float, float]
# Ключевая точка позы: (x, y, видимость 0..1).
Keypoint = tuple[float, float, float]
# (ширина, высота) кадра в пикселях.
FrameSize = tuple[int, int]


class Kp:
    """Индексы ключевых точек COCO (формат YOLOv8-pose)."""

    NOSE = 0
    LEFT_EYE, RIGHT_EYE = 1, 2
    LEFT_EAR, RIGHT_EAR = 3, 4
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
    LEFT_HIP, RIGHT_HIP = 11, 12
    LEFT_ANKLE, RIGHT_ANKLE = 15, 16

    HEAD = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)
    SHOULDERS = (LEFT_SHOULDER, RIGHT_SHOULDER)
    HIPS = (LEFT_HIP, RIGHT_HIP)
    TORSO = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    ANKLES = (LEFT_ANKLE, RIGHT_ANKLE)


@dataclass(frozen=True)
class PPEItem:
    """Найденное детектором средство защиты."""

    class_name: str
    box: Box
    confidence: float


@dataclass
class Person:
    """Человек на кадре. `ppe` заполняется на этапе привязки."""

    track_id: int
    box: Box
    confidence: float
    keypoints: tuple[Keypoint, ...] | None = None
    ppe: list[PPEItem] = field(default_factory=list)

    @property
    def present_classes(self) -> frozenset[str]:
        return frozenset(item.class_name for item in self.ppe)


class Status(Enum):
    """Итог проверки человека на одном кадре."""

    OK = "ok"
    VIOLATION = "violation"
    UNKNOWN = "unknown"  # нужная часть тела не видна — судить нельзя


@dataclass(frozen=True)
class PersonStatus:
    track_id: int
    status: Status
    missing: frozenset[str] = frozenset()
    undetermined: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Zone:
    """Запретная зона. Координаты нормированы к размеру кадра (0..1),
    поэтому зона не «съезжает» при изменении разрешения."""

    name: str
    polygon: tuple[Point, ...]


class Rule:
    """Имена правил. Используются как часть ключа состояния и в событиях."""

    PPE = "ppe"
    ZONE = "zone"


class ObsState(Enum):
    """Что видно на этом кадре по конкретному правилу у конкретного человека."""

    VIOLATING = "violating"  # нарушение видно (СИЗ нет / человек в зоне)
    COMPLIANT = "compliant"  # всё в порядке (СИЗ есть / человек вне зоны)
    UNKNOWN = "unknown"  # судить нельзя — таймеры гистерезиса стоят


@dataclass(frozen=True)
class Observation:
    """Наблюдение одного правила за одним человеком на одном кадре.

    Ключ (track_id, rule, subject): у одного человека отдельно ведутся
    каска, жилет и каждая зона — нарушения не мешают друг другу.
    """

    track_id: int
    rule: str
    subject: str  # класс СИЗ ("helmet") или имя зоны ("crane")
    state: ObsState
    box: Box
    confidence: float

    @property
    def key(self) -> tuple[int, str, str]:
        return (self.track_id, self.rule, self.subject)


def violation_type(rule: str, subject: str) -> str:
    """Строковый тип нарушения: 'helmet', 'vest', 'zone:crane'."""
    return f"zone:{subject}" if rule == Rule.ZONE else subject


class EpisodeKind(Enum):
    OPENED = "opened"  # нарушение подтверждено — срочное оповещение
    UPDATED = "updated"  # нарушение продолжается (напоминание)
    CLOSED = "closed"  # нарушение завершено


class CloseReason(Enum):
    RESOLVED = "resolved"  # СИЗ надето / человек вышел из зоны
    TRACK_LOST = "track_lost"  # человек пропал из кадра дольше lost_ttl
    STREAM_ENDED = "stream_ended"  # конец видео или остановка программы


@dataclass(frozen=True)
class EpisodeEvent:
    """Событие жизненного цикла эпизода нарушения. Время — секунды кадра."""

    kind: EpisodeKind
    episode_id: str
    camera_id: str
    rule: str
    subject: str
    track_id: int  # текущий id (после ID switch — новый)
    track_ids: tuple[int, ...]  # все id, под которыми человек был в эпизоде
    started_ts: float  # первое наблюдение нарушения в окне подтверждения
    event_ts: float  # кадр, на котором произошло событие
    box: Box  # последний известный бокс человека
    confidence: float  # максимальная уверенность детекции человека
    violating_sec: float  # сколько секунд нарушение реально наблюдалось
    ended_ts: float | None = None  # только для CLOSED
    close_reason: CloseReason | None = None

    @property
    def violation_type(self) -> str:
        return violation_type(self.rule, self.subject)

    @property
    def duration_sec(self) -> float | None:
        return None if self.ended_ts is None else self.ended_ts - self.started_ts
