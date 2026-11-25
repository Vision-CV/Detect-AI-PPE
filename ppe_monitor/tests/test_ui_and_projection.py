"""Отрисовка, окно предпросмотра, проекция на пол, раздача пакетов (Broadcast)."""

import cv2
import numpy as np
import pytest

from ppe_monitor.domain.analyzer import Analyzer
from ppe_monitor.domain.models import Zone
from ppe_monitor.domain.policy import PPEPolicy
from ppe_monitor.pipeline.projection import FloorProjector
from ppe_monitor.sinks.base import Broadcast
from ppe_monitor.ui import display as display_module
from ppe_monitor.ui.renderer import COLOR_CONFIRMED, COLOR_OK, COLOR_PENDING, draw, encode_jpeg

from factories import helmet_on, person, vest_on

IMAGE = np.zeros((300, 400, 3), np.uint8)
WORKER = person(1, (100, 40, 180, 280))


def analysis(*items, zones=()):
    return Analyzer(PPEPolicy(), zones).analyze([WORKER], list(items), (400, 300))


def edge_color(img, box=WORKER.box):
    """Цвет рамки человека — пиксель на левой границе бокса посередине высоты."""
    x1, y1, _, y2 = map(int, box)
    return tuple(int(v) for v in img[(y1 + y2) // 2, x1])


# ------------------------------------------------------------------ renderer


def test_draw_does_not_modify_input():
    original = IMAGE.copy()
    out = draw(IMAGE, analysis())
    assert np.array_equal(IMAGE, original)
    assert out.shape == IMAGE.shape and not np.array_equal(out, IMAGE)


@pytest.mark.parametrize(
    "items,confirmed,color",
    [
        ((helmet_on(WORKER), vest_on(WORKER)), set(), COLOR_OK),  # всё надето
        ((vest_on(WORKER),), set(), COLOR_PENDING),  # нет каски, ещё не подтверждено
        ((vest_on(WORKER),), {1}, COLOR_CONFIRMED),  # подтверждённое нарушение
    ],
)
def test_person_frame_color(items, confirmed, color):
    out = draw(IMAGE, analysis(*items), confirmed_track_ids=confirmed)
    assert edge_color(out) == color


def test_zones_are_drawn():
    zone = Zone("crane", ((0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)))
    out = draw(IMAGE, analysis(helmet_on(WORKER), vest_on(WORKER)), zones=[zone])
    assert out[280, 350].any()  # внутри зоны — полупрозрачная заливка
    assert not out[20, 20].any()  # вне зоны и людей — без изменений


def test_encode_jpeg_roundtrip():
    data = encode_jpeg(np.full((50, 60, 3), 128, np.uint8), quality=95)
    assert data[:2] == b"\xff\xd8"
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (50, 60, 3) and abs(int(decoded.mean()) - 128) <= 2


# ------------------------------------------------------------------ окно


class FakeCV2:
    def __init__(self, key=-1, visible=1.0):
        self.key, self.visible, self.shown = key, visible, 0
        self.WINDOW_NORMAL, self.WND_PROP_VISIBLE = 0, 4

    def namedWindow(self, *a):  # noqa: N802
        pass

    def imshow(self, name, img):
        self.shown += 1

    def waitKey(self, ms):  # noqa: N802
        return self.key

    def getWindowProperty(self, name, prop):  # noqa: N802
        return self.visible

    def destroyWindow(self, name):  # noqa: N802
        pass


@pytest.mark.parametrize("key,visible,expected", [(-1, 1.0, True), (ord("q"), 1.0, False), (27, 1.0, False), (-1, 0.0, False)])
def test_window_show(monkeypatch, key, visible, expected):
    fake = FakeCV2(key, visible)
    monkeypatch.setattr(display_module, "cv2", fake)
    window = display_module.Window("w")
    assert window.show(IMAGE) is expected
    assert fake.shown == 1
    window.close()


# ------------------------------------------------------------------ проекция


def test_projector_maps_calibration_points():
    image_pts = [[100, 400], [500, 400], [450, 200], [150, 200]]
    world_pts = [[0, 0], [4, 0], [4, 10], [0, 10]]
    proj = FloorProjector(image_pts, world_pts)
    for (px, py), (wx, wz) in zip(image_pts, world_pts):
        x, z = proj.to_floor(px, py)
        assert (x, z) == (pytest.approx(wx, abs=1e-3), pytest.approx(wz, abs=1e-3))
    mid_x, _ = proj.to_floor(300, 400)
    assert mid_x == pytest.approx(2.0, abs=1e-3)  # середина нижней кромки — 2 м


def test_projector_needs_four_points():
    with pytest.raises(ValueError):
        FloorProjector([[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 0], [1, 1]])


# ------------------------------------------------------------------ Broadcast


class Recording:
    def __init__(self, fail=False):
        self.got, self.closed, self.fail = [], False, fail

    def publish(self, package):
        if self.fail:
            raise OSError("диск полон")
        self.got.append(package)

    def close(self):
        self.closed = True
        if self.fail:
            raise OSError("close")


def test_broadcast_isolates_failing_sink():
    bad, good = Recording(fail=True), Recording()
    b = Broadcast([bad, good])
    b.publish("pkg")
    b.close()
    assert good.got == ["pkg"] and good.closed and bad.closed
