from dataclasses import replace

from ppe_monitor.domain.compliance import evaluate, ppe_observations
from ppe_monitor.domain.models import ObsState, Status
from ppe_monitor.domain.policy import PPEPolicy

from factories import HIDDEN, helmet_on, person, vest_on

POLICY = PPEPolicy()


def with_ppe(p, *items):
    return replace(p, ppe=list(items))


def test_fully_equipped_is_ok():
    p = person()
    status = evaluate(with_ppe(p, helmet_on(p), vest_on(p)), POLICY)
    assert status.status is Status.OK
    assert {o.state for o in ppe_observations(p, status, POLICY)} == {ObsState.COMPLIANT}


def test_missing_helmet_is_violation():
    p = person()
    status = evaluate(with_ppe(p, vest_on(p)), POLICY)
    assert status.status is Status.VIOLATION
    assert status.missing == {"helmet"}


def test_missing_both():
    p = person()
    status = evaluate(p, POLICY)
    assert status.missing == {"helmet", "vest"}


def test_head_cut_by_frame_edge_is_unknown_not_violation():
    p = person(box=(100, 0, 200, 300))
    status = evaluate(with_ppe(p, vest_on(p)), POLICY)
    assert status.status is Status.UNKNOWN
    assert status.undetermined == {"helmet"}


def test_hidden_torso_makes_vest_undetermined():
    p = person(torso=HIDDEN, head=0.9)
    status = evaluate(with_ppe(p, helmet_on(p)), POLICY)
    assert status.status is Status.UNKNOWN
    assert status.undetermined == {"vest"}


def test_visible_part_still_reported_when_other_is_hidden():
    p = person(torso=HIDDEN)
    status = evaluate(p, POLICY)
    assert status.status is Status.VIOLATION
    assert status.missing == {"helmet"}
    assert status.undetermined == {"vest"}


def test_without_keypoints_assume_visible():
    p = person(keypoints=False)
    assert evaluate(p, POLICY).status is Status.VIOLATION


def test_one_observation_per_required_class():
    p = person(torso=HIDDEN)
    status = evaluate(with_ppe(p, helmet_on(p)), POLICY)
    states = {o.subject: o.state for o in ppe_observations(p, status, POLICY)}
    assert states == {"helmet": ObsState.COMPLIANT, "vest": ObsState.UNKNOWN}
