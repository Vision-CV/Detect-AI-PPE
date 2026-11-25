import pytest

from ppe_monitor.domain.geometry import iou, point_in_polygon, visible_points

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
# Г-образный (невыпуклый) многоугольник.
L_SHAPE = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]


@pytest.mark.parametrize(
    ("point", "polygon", "expected"),
    [
        ((5, 5), SQUARE, True),
        ((15, 5), SQUARE, False),
        ((-1, 5), SQUARE, False),
        ((2, 8), L_SHAPE, True),
        ((8, 8), L_SHAPE, False),  # в «вырезе» буквы Г
    ],
)
def test_point_in_polygon(point, polygon, expected):
    assert point_in_polygon(point, polygon) is expected


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_visible_points_filters_by_visibility():
    kps = [(1, 1, 0.9), (2, 2, 0.1), (3, 3, 0.5)]
    assert visible_points(kps, [0, 1, 2], 0.3) == [(1, 1), (3, 3)]
    assert visible_points(None, [0], 0.3) == []
