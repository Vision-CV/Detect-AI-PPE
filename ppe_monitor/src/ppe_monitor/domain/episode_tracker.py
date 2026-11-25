"""Гистерезис: покадровые наблюдения → эпизоды нарушений (открыт / закрыт).

Для каждого ключа (человек, правило, предмет) — отдельный автомат:

    NORMAL ──[нарушение ≥ enter_sec и доля ≥ enter_ratio]──► VIOLATION   → OPENED
    VIOLATION ──[всё в порядке непрерывно ≥ exit_sec]──► NORMAL          → CLOSED(resolved)
    VIOLATION ──[человек не виден дольше lost_ttl_sec]──► удалён         → CLOSED(track_lost)

Почему так:
- Вход и выход разные: войти в нарушение можно за 3 с, а выйти — только
  когда СИЗ надёжно видно 5 с. Мигание детектора не закрывает нарушение и
  не плодит повторные оповещения.
- Всё в секундах, а не в кадрах: промежуток между соседними кадрами
  засчитывается состоянию, которое наблюдалось в начале промежутка. Падение
  FPS или пропуск кадров не меняют смысл порогов. Промежуток длиннее
  max_gap_sec (человек пропадал из кадра) считается «не видно».
- UNKNOWN (часть тела не видна) не добавляет времени ни «нарушению», ни
  «порядку» — таймеры просто стоят. «Не вижу» ≠ «нет» и ≠ «есть».
- Эпизод переживает кратковременную потерю трека, а при ID switch
  (тот же человек получил новый id рядом с тем же местом) — переходит
  к новому id, а не открывается заново.

Время передаётся параметром, поэтому логика тестируется мгновенно.
"""

from __future__ import annotations

import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from ppe_monitor.domain.geometry import box_center, box_height, distance
from ppe_monitor.domain.models import (
    Box,
    CloseReason,
    EpisodeEvent,
    EpisodeKind,
    Observation,
    ObsState,
)

Key = tuple[int, str, str]


@dataclass(frozen=True)
class Timing:
    enter_sec: float = 3.0  # сколько наблюдаемого времени нужно для подтверждения
    exit_sec: float = 5.0  # сколько непрерывного «порядка» нужно для снятия
    enter_ratio: float = 0.6  # доля «нарушение» в окне enter_sec
    lost_ttl_sec: float = 30.0  # сколько помнить пропавшего человека
    reminder_sec: float = 0.0  # повторное оповещение, пока нарушение длится (0 — выкл.)
    max_gap_sec: float = 2.0  # промежуток длиннее — «не видно» (должен быть > интервала кадров)


@dataclass
class _Episode:
    episode_id: str
    started_ts: float
    opened_ts: float
    track_ids: list[int]
    last_alert_ts: float
    compliant_streak: float = 0.0
    violating_sec: float = 0.0
    max_confidence: float = 0.0


@dataclass
class _State:
    rule: str
    subject: str
    track_id: int
    box: Box
    last_obs_ts: float  # последний кадр с любым наблюдением (в т.ч. UNKNOWN)
    last_state: ObsState | None = None  # что было видно на том кадре
    # Окно доли: (время, секунды, нарушение?) за последние ~enter_sec наблюдаемого времени.
    window: deque[tuple[float, float, bool]] = field(default_factory=deque)
    window_total: float = 0.0
    window_violating: float = 0.0
    # Кандидат в нарушение: когда впервые увидели и сколько наблюдаемых секунд прошло.
    pending_since: float | None = None
    pending_observed: float = 0.0
    episode: _Episode | None = None

    @property
    def key(self) -> Key:
        return (self.track_id, self.rule, self.subject)

    def reset_window(self) -> None:
        self.window.clear()
        self.window_total = 0.0
        self.window_violating = 0.0
        self.pending_since = None
        self.pending_observed = 0.0


class EpisodeTracker:
    def __init__(
        self,
        camera_id: str,
        timings: Mapping[str, Timing],
        *,
        reid_max_distance: float = 1.0,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        """reid_max_distance — насколько далеко (в ростах человека) может
        появиться новый трек, чтобы унаследовать открытый эпизод пропавшего."""
        self._camera_id = camera_id
        self._timings = dict(timings)
        self._reid = reid_max_distance
        self._new_id = id_factory
        self._states: dict[Key, _State] = {}
        self._last_ts: float | None = None
        self._seen_now: set[int] = set()

    # ------------------------------------------------------------ публичное

    def update(self, ts: float, observations: Iterable[Observation]) -> list[EpisodeEvent]:
        """Учитывает наблюдения кадра со временем `ts`, возвращает события."""
        observations = list(observations)
        self._last_ts = ts
        self._seen_now = {o.track_id for o in observations}
        events: list[EpisodeEvent] = []

        for obs in observations:
            state = self._states.get(obs.key) or self._inherit_or_create(obs, ts)
            event = self._observe(state, obs, ts)
            if event is not None:
                events.append(event)

        events.extend(self._expire(ts))
        return events

    def flush(self, reason: CloseReason = CloseReason.STREAM_ENDED) -> list[EpisodeEvent]:
        """Закрывает все открытые эпизоды (конец видео, остановка программы)."""
        if self._last_ts is None:
            return []
        events = [
            self._closed(state, self._last_ts, min(state.last_obs_ts, self._last_ts), reason)
            for state in self._states.values()
            if state.episode is not None
        ]
        self._states.clear()
        return events

    def open_episodes(self) -> dict[str, Key]:
        """Открытые эпизоды, чей человек виден на последнем кадре:
        episode_id -> (текущий track_id, правило, предмет)."""
        return {
            s.episode.episode_id: s.key
            for s in self._states.values()
            if s.episode is not None and s.track_id in self._seen_now
        }

    def open_track_ids(self) -> set[int]:
        """Люди с открытым нарушением, видимые на последнем кадре (для отрисовки)."""
        return {key[0] for key in self.open_episodes().values()}

    # ------------------------------------------------------------ автомат

    def _observe(self, state: _State, obs: Observation, ts: float) -> EpisodeEvent | None:
        timing = self._timings[state.rule]
        gap = ts - state.last_obs_ts
        previous = state.last_state
        state.last_obs_ts = ts
        state.last_state = obs.state
        state.box = obs.box
        state.track_id = obs.track_id

        # Промежуток с прошлого кадра — тому, что тогда было видно. UNKNOWN и
        # длинные пропуски (человека не было в кадре) не засчитываются никуда.
        if 0 < gap <= timing.max_gap_sec and previous not in (None, ObsState.UNKNOWN):
            self._account(state, ts, gap, previous is ObsState.VIOLATING, timing)

        ep = state.episode
        if ep is not None and ep.compliant_streak >= timing.exit_sec:
            event = self._closed(state, ts, ts - ep.compliant_streak, CloseReason.RESOLVED)
            state.episode = None
            state.reset_window()  # новое нарушение снова потребует полных enter_sec
            if obs.state is ObsState.VIOLATING:
                # СИЗ было 5 с и снова снято — сразу начинаем новый отсчёт.
                state.pending_since = ts
            return event

        if obs.state is not ObsState.VIOLATING:
            return None
        if ep is None:
            if state.pending_since is None:
                state.pending_since = ts
            if self._should_open(state, timing):
                return self._open(state, obs, ts)
            return None

        ep.max_confidence = max(ep.max_confidence, obs.confidence)
        if timing.reminder_sec and ts - ep.last_alert_ts >= timing.reminder_sec:
            ep.last_alert_ts = ts
            return self._event(EpisodeKind.UPDATED, state, ts)
        return None

    @staticmethod
    def _account(state: _State, ts: float, dt: float, violating: bool, timing: Timing) -> None:
        state.window.append((ts, dt, violating))
        state.window_total += dt
        state.window_violating += dt if violating else 0.0
        # Окно — минимальный хвост, покрывающий enter_sec наблюдаемого времени.
        while state.window and state.window_total - state.window[0][1] >= timing.enter_sec:
            _, old_dt, old_violating = state.window.popleft()
            state.window_total -= old_dt
            state.window_violating -= old_dt if old_violating else 0.0

        if state.pending_since is not None:
            state.pending_observed += dt
            if state.window_violating <= 1e-9:
                state.pending_since = None  # за целое окно ни одного нарушения — кандидат снят
                state.pending_observed = 0.0

        ep = state.episode
        if ep is not None:
            if violating:
                ep.compliant_streak = 0.0
                ep.violating_sec += dt
            else:
                ep.compliant_streak += dt

    @staticmethod
    def _should_open(state: _State, timing: Timing) -> bool:
        eps = 1e-6
        return (
            state.pending_observed + eps >= timing.enter_sec
            and state.window_total + eps >= timing.enter_sec
            and state.window_violating + eps >= timing.enter_ratio * state.window_total
        )

    def _open(self, state: _State, obs: Observation, ts: float) -> EpisodeEvent:
        state.episode = _Episode(
            episode_id=self._new_id(),
            started_ts=state.pending_since if state.pending_since is not None else ts,
            opened_ts=ts,
            track_ids=[obs.track_id],
            last_alert_ts=ts,
            violating_sec=state.window_violating,
            max_confidence=obs.confidence,
        )
        return self._event(EpisodeKind.OPENED, state, ts)

    def _expire(self, ts: float) -> list[EpisodeEvent]:
        events = []
        for key, state in list(self._states.items()):
            if state.track_id in self._seen_now:
                continue
            if ts - state.last_obs_ts > self._timings[state.rule].lost_ttl_sec:
                if state.episode is not None:
                    events.append(self._closed(state, ts, state.last_obs_ts, CloseReason.TRACK_LOST))
                del self._states[key]
        return events

    # ------------------------------------------------------------ ID switch

    def _inherit_or_create(self, obs: Observation, ts: float) -> _State:
        donor = self._find_donor(obs, ts)
        if donor is not None:
            del self._states[donor.key]
            donor.track_id = obs.track_id
            donor.episode.track_ids.append(obs.track_id)
            # Время пропажи не должно засчитаться ни в нарушение, ни в порядок.
            donor.last_obs_ts = ts
            donor.last_state = None
            self._states[donor.key] = donor
            return donor
        state = _State(obs.rule, obs.subject, obs.track_id, obs.box, last_obs_ts=ts)
        self._states[state.key] = state
        return state

    def _find_donor(self, obs: Observation, ts: float) -> _State | None:
        """Открытый эпизод того же правила у пропавшего трека рядом с новым."""
        height = box_height(obs.box)
        center = box_center(obs.box)
        best, best_dist = None, None
        for state in self._states.values():
            if (
                state.episode is None
                or state.rule != obs.rule
                or state.subject != obs.subject
                or state.track_id in self._seen_now  # этот человек ещё в кадре
                or ts - state.last_obs_ts > self._timings[state.rule].lost_ttl_sec
            ):
                continue
            d = distance(center, box_center(state.box))
            if d <= self._reid * height and (best_dist is None or d < best_dist):
                best, best_dist = state, d
        return best

    # ------------------------------------------------------------ события

    def _event(self, kind: EpisodeKind, state: _State, ts: float, **extra) -> EpisodeEvent:
        ep = state.episode
        return EpisodeEvent(
            kind=kind,
            episode_id=ep.episode_id,
            camera_id=self._camera_id,
            rule=state.rule,
            subject=state.subject,
            track_id=state.track_id,
            track_ids=tuple(ep.track_ids),
            started_ts=ep.started_ts,
            event_ts=ts,
            box=state.box,
            confidence=ep.max_confidence,
            violating_sec=round(ep.violating_sec, 3),
            **extra,
        )

    def _closed(self, state: _State, ts: float, ended_ts: float, reason: CloseReason) -> EpisodeEvent:
        return self._event(EpisodeKind.CLOSED, state, ts, ended_ts=ended_ts, close_reason=reason)
