"""Гистерезис-автомат. Сценарии S1–S6 из аудита — регрессионные тесты."""

import itertools
import math

import pytest

from ppe_monitor.domain.episode_tracker import EpisodeTracker, Timing
from ppe_monitor.domain.models import CloseReason, EpisodeKind, Observation, ObsState, Rule

PPE_T = Timing(enter_sec=3.0, exit_sec=5.0, enter_ratio=0.6, lost_ttl_sec=30.0, max_gap_sec=2.0)
ZONE_T = Timing(enter_sec=1.0, exit_sec=3.0, enter_ratio=0.6, lost_ttl_sec=30.0, max_gap_sec=2.0)

V, C, U = ObsState.VIOLATING, ObsState.COMPLIANT, ObsState.UNKNOWN
BOX = (100.0, 100.0, 200.0, 400.0)  # рост 300 px


def make_tracker(**kw):
    counter = itertools.count(1)
    return EpisodeTracker(
        "cam1", {Rule.PPE: PPE_T, Rule.ZONE: ZONE_T}, id_factory=lambda: f"ep{next(counter)}", **kw
    )


def obs(state, track_id=1, subject="helmet", rule=Rule.PPE, box=BOX):
    return Observation(track_id, rule, subject, state, box, 0.9)


def play(tracker, timeline, fps=10.0, t0=0.0):
    """timeline: список (длительность_сек, [наблюдения] | функция) → все события."""
    events, n = [], 0
    for seconds, frame_obs in timeline:
        for _ in range(math.floor(seconds * fps + 0.5)):
            events += tracker.update(round(t0 + n / fps, 6), frame_obs)
            n += 1
    return events


def kinds(events):
    return [(e.kind.value, e.subject, round(e.event_ts, 1)) for e in events]


# ------------------------------------------------------------- вход 3 с / выход 5 с


def test_opens_after_3_seconds_of_violation():
    t = make_tracker()
    events = play(t, [(3.05, [obs(V)])])
    assert kinds(events) == [("opened", "helmet", 3.0)]
    assert events[0].started_ts == 0.0


@pytest.mark.parametrize("fps", [1.0, 5.0, 30.0])
def test_enter_time_does_not_depend_on_fps(fps):
    events = play(make_tracker(), [(3.0 + 1 / fps, [obs(V)])], fps=fps)
    assert [round(e.event_ts, 2) for e in events] == [3.0]


def test_closes_5_seconds_after_ppe_put_on():
    t = make_tracker()
    events = play(t, [(10, [obs(V)]), (6, [obs(C)])])
    opened, closed = events
    assert closed.kind is EpisodeKind.CLOSED
    assert closed.close_reason is CloseReason.RESOLVED
    assert closed.event_ts == pytest.approx(15.0)  # каска надета на 10.0 → +5 с
    assert closed.ended_ts == pytest.approx(10.0)  # нарушение фактически кончилось на 10.0
    assert closed.duration_sec == pytest.approx(10.0)


def test_s1_helmet_on_off_cycle_gives_single_episode():
    """S1: 4 с без каски / 2.5 с в каске, минуту. Раньше — 9 оповещений."""
    events = play(make_tracker(), [(4, [obs(V)]), (2.5, [obs(C)])] * 10)
    assert kinds(events) == [("opened", "helmet", 3.0)]


def test_new_violation_after_real_fix_is_reported_again():
    # Каска надета 6 с (эпизод закрыт), потом снова снята — это НОВОЕ нарушение.
    events = play(make_tracker(), [(4, [obs(V)]), (6, [obs(C)]), (4, [obs(V)])])
    assert [e.kind.value for e in events] == ["opened", "closed", "opened"]
    assert events[0].episode_id != events[2].episode_id


def test_s5_detector_flicker_does_not_open():
    """S5: нарушение видно на 3 кадрах из 39 (8%). Раньше — подтверждалось."""
    timeline = [(0.1, [obs(V)] if i in (0, 19, 38) else [obs(C)]) for i in range(40)]
    assert play(make_tracker(), timeline) == []


def test_mostly_violating_with_noise_still_opens():
    # 75% кадров без каски (детектор иногда «видит» её) — нарушение должно открыться.
    timeline = [(0.1, [obs(C)] if i % 4 == 0 else [obs(V)]) for i in range(50)]
    assert [e.kind.value for e in play(make_tracker(), timeline)] == ["opened"]


def test_short_helmet_glimpse_does_not_close():
    events = play(make_tracker(), [(4, [obs(V)]), (4.9, [obs(C)]), (1, [obs(V)]), (4.9, [obs(C)])])
    assert [e.kind.value for e in events] == ["opened"]


# ------------------------------------------------------------- UNKNOWN замораживает


def test_unknown_freezes_entry_timer():
    # 2 с нарушения, 10 с «не видно», ещё 1 с нарушения → наблюдали 3 с.
    events = play(make_tracker(), [(2, [obs(V)]), (10, [obs(U)]), (1.05, [obs(V)])])
    assert [round(e.event_ts, 1) for e in events] == [13.0]


def test_unknown_freezes_exit_timer():
    # В эпизоде: 3 с в каске, 10 с не видно, 2 с в каске → итого 5 с порядка.
    events = play(make_tracker(), [(4, [obs(V)]), (3, [obs(C)]), (10, [obs(U)]), (2.05, [obs(C)])])
    assert [e.kind.value for e in events] == ["opened", "closed"]
    assert events[1].event_ts == pytest.approx(19.0, abs=0.11)


def test_unknown_alone_never_opens():
    assert play(make_tracker(), [(60, [obs(U)])]) == []


# ------------------------------------------------------------- раздельно по СИЗ


def test_s4_each_ppe_class_has_own_episode():
    """S4: сначала нет каски, через 4 с снят и жилет. Раньше про жилет молчали."""
    events = play(
        make_tracker(),
        [
            (4, [obs(V, subject="helmet"), obs(C, subject="vest")]),
            (4, [obs(V, subject="helmet"), obs(V, subject="vest")]),
        ],
    )
    assert kinds(events) == [("opened", "helmet", 3.0), ("opened", "vest", 7.0)]


def test_rules_have_own_timings():
    events = play(make_tracker(), [(3.05, [obs(V), obs(V, rule=Rule.ZONE, subject="crane")])])
    assert kinds(events) == [("opened", "crane", 1.0), ("opened", "helmet", 3.0)]


def test_people_are_independent():
    events = play(make_tracker(), [(2, [obs(V, 1)]), (3.05, [obs(V, 1), obs(V, 2, box=(500, 100, 600, 400))])])
    assert [(e.track_id, round(e.event_ts, 1)) for e in events] == [(1, 3.0), (2, 5.0)]


# ------------------------------------------------------------- потеря трека, ID switch


def test_s2_short_exit_from_frame_keeps_episode():
    """S2: ушёл из кадра на 3 с и вернулся. Раньше — повторное оповещение."""
    events = play(make_tracker(), [(4, [obs(V)]), (3, []), (4, [obs(V)])])
    assert [e.kind.value for e in events] == ["opened"]


def test_gap_does_not_count_as_violation_time():
    # 1 с нарушения, пропал на 20 с, вернулся на 0.5 с: время пропажи не засчитано.
    events = play(make_tracker(), [(1, [obs(V)]), (20, []), (0.5, [obs(V)])])
    assert events == []


def test_long_loss_closes_episode_as_track_lost():
    events = play(make_tracker(), [(4, [obs(V)]), (31, [])])
    assert [e.kind.value for e in events] == ["opened", "closed"]
    assert events[1].close_reason is CloseReason.TRACK_LOST
    assert events[1].ended_ts == pytest.approx(3.9)


def test_s3_id_switch_inherits_episode():
    """S3: тот же человек получил новый id. Раньше — второе оповещение."""
    moved = (110.0, 100.0, 210.0, 400.0)
    events = play(make_tracker(), [(4, [obs(V, 1)]), (0.5, []), (6, [obs(V, 7, box=moved)])])
    assert [e.kind.value for e in events] == ["opened"]

    t = make_tracker()
    play(t, [(4, [obs(V, 1)]), (0.5, []), (1, [obs(V, 7, box=moved)])])
    [closed] = play(t, [(5.05, [obs(C, 7, box=moved)])], t0=5.5)
    assert closed.track_ids == (1, 7)
    assert closed.close_reason is CloseReason.RESOLVED


def test_far_new_track_does_not_inherit():
    far = (900.0, 100.0, 1000.0, 400.0)  # 800 px при росте 300 px
    events = play(make_tracker(), [(4, [obs(V, 1)]), (0.5, []), (3.05, [obs(V, 7, box=far)])])
    assert [(e.kind.value, e.track_id) for e in events] == [("opened", 1), ("opened", 7)]


def test_visible_person_episode_is_not_stolen():
    # Человек 1 всё ещё в кадре — новый человек 2 рядом не забирает его эпизод.
    near = (120.0, 100.0, 220.0, 400.0)
    events = play(make_tracker(), [(4, [obs(V, 1)]), (3.05, [obs(V, 1), obs(V, 2, box=near)])])
    assert [(e.kind.value, e.track_id) for e in events] == [("opened", 1), ("opened", 2)]


# ------------------------------------------------------------- прочее


def test_reminder_while_violation_lasts():
    t = EpisodeTracker("c", {Rule.PPE: Timing(reminder_sec=10.0)})
    events = play(t, [(25.05, [obs(V)])])
    assert [(e.kind.value, round(e.event_ts)) for e in events] == [("opened", 3), ("updated", 13), ("updated", 23)]


def test_flush_closes_open_episodes():
    t = make_tracker()
    play(t, [(4, [obs(V)])])
    [closed] = t.flush()
    assert closed.close_reason is CloseReason.STREAM_ENDED
    assert t.flush() == []


def test_open_track_ids_for_rendering():
    t = make_tracker()
    play(t, [(2, [obs(V)])])
    assert t.open_track_ids() == set()
    play(t, [(1.5, [obs(V)])], t0=2.0)
    assert t.open_track_ids() == {1}
