from ppe_monitor.domain.models import ObsState, Rule, Zone
from ppe_monitor.domain.zones import foot_point, zone_observations

from factories import HIDDEN, person

FRAME = (1000, 500)
# Правая половина кадра.
RIGHT_HALF = Zone("crane", ((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)))


def findings(*persons):
    """Только наблюдения «в зоне» (как раньше возвращала zone_findings)."""
    return [o for o in states(*persons) if o.state is ObsState.VIOLATING]


def states(*persons):
    return zone_observations(persons, [RIGHT_HALF], FRAME, 0.3, 4)


def test_person_standing_in_zone():
    [f] = findings(person(1, (700, 100, 800, 400)))
    assert f.rule == Rule.ZONE
    assert f.subject == "crane"
    assert f.track_id == 1


def test_person_outside_zone_is_compliant():
    [o] = states(person(1, (100, 100, 200, 400)))
    assert o.state is ObsState.COMPLIANT


def test_zone_is_checked_by_feet_not_by_box():
    # Бокс заходит в зону, но ноги — снаружи (например, человек наклонился).
    p = person(1, (400, 100, 600, 400))
    p.keypoints = tuple(
        (430.0, 395.0, kp[2]) if idx in (15, 16) else kp for idx, kp in enumerate(p.keypoints)
    )
    assert findings(p) == []


def test_feet_cut_by_frame_bottom_give_no_decision():
    p = person(1, (700, 100, 800, 500), ankles=HIDDEN)
    assert foot_point(p, FRAME, 0.3, 4) is None
    [o] = states(p)
    assert o.state is ObsState.UNKNOWN


def test_no_zones_no_findings():
    assert zone_observations([person()], [], FRAME, 0.3, 4) == []
