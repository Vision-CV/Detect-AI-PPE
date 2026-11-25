from ppe_monitor.domain.association import associate
from ppe_monitor.domain.models import PPEItem
from ppe_monitor.domain.policy import PPEPolicy

from factories import helmet_in_hand, helmet_on, person, vest_on

POLICY = PPEPolicy()


def classes(p):
    return sorted(item.class_name for item in p.ppe)


def test_worn_helmet_and_vest_are_assigned():
    p = person()
    [result] = associate([p], [helmet_on(p), vest_on(p)], POLICY)
    assert classes(result) == ["helmet", "vest"]


def test_input_persons_are_not_mutated():
    p = person()
    associate([p], [helmet_on(p)], POLICY)
    assert p.ppe == []


def test_helmet_in_hand_is_not_counted_as_worn():
    p = person()
    [result] = associate([p], [helmet_in_hand(p)], POLICY)
    assert result.ppe == []


def test_helmet_in_hand_without_keypoints_is_not_counted():
    p = person(keypoints=False)
    [result] = associate([p], [helmet_in_hand(p)], POLICY)
    assert result.ppe == []


def test_one_helmet_goes_to_one_person_only():
    front = person(1, (100, 100, 220, 400))
    back = person(2, (130, 90, 230, 380))  # частично перекрыт первым
    helmet = helmet_on(back)
    result = associate([front, back], [helmet], POLICY)
    owners = [p.track_id for p in result if p.ppe]
    assert owners == [2]


def test_person_gets_at_most_one_helmet():
    p = person()
    h1 = helmet_on(p, conf=0.9)
    h2 = PPEItem("helmet", h1.box, 0.6)
    [result] = associate([p], [h1, h2], POLICY)
    assert result.ppe == [h1]  # остаётся более уверенная


def test_oversized_vest_is_rejected():
    p = person()
    huge = PPEItem("vest", (0, 150, 600, 300), 0.9)  # в 6 раз шире человека
    [result] = associate([p], [huge], POLICY)
    assert result.ppe == []


def test_unknown_class_attaches_to_whole_body():
    p = person()
    glove = PPEItem("gloves", (190, 250, 210, 270), 0.7)
    [result] = associate([p], [glove], POLICY)
    assert classes(result) == ["gloves"]
