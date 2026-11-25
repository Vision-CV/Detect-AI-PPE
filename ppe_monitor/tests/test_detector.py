"""YoloDetector: разбор результатов ultralytics (на подменной модели) и реальные модели (slow)."""

import sys
import types

import numpy as np
import pytest
import torch

from ppe_monitor.pipeline.detector import YoloDetector

from conftest import VIDEOS, WEIGHTS, require_real_assets


class FakeResult:
    def __init__(self, boxes, ids=None, cls=None, keypoints=None):
        n = len(boxes)
        self.boxes = types.SimpleNamespace(
            xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(n, 4),
            conf=torch.full((n,), 0.9),
            cls=torch.tensor(cls if cls is not None else [0] * n),
            id=None if ids is None else torch.tensor(ids, dtype=torch.float32),
        )
        self.keypoints = None if keypoints is None else types.SimpleNamespace(data=torch.tensor(keypoints))


class FakeYOLO:
    instances = []

    def __init__(self, path):
        self.path = path
        self.calls = []
        self.names = {0: "person"} if "pose" in path else {0: "helmet", 1: "human", 2: "vest"}
        self.result = None
        FakeYOLO.instances.append(self)

    def track(self, image, **kwargs):
        self.calls.append(("track", kwargs))
        return [self.result]

    def __call__(self, image, **kwargs):
        self.calls.append(("predict", kwargs))
        return [self.result]


@pytest.fixture
def make_detector(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    FakeYOLO.instances = []

    def factory(ppe_name="ppe.pt", half=False):
        for name in ("pose.pt", ppe_name):
            (tmp_path / name).write_bytes(b"")
        det = YoloDetector(
            tmp_path / "pose.pt", tmp_path / ppe_name, device="cpu", imgsz=640, half=half,
            person_conf=0.5, ppe_conf=0.35, iou=0.5, tracker="bytetrack.yaml", ignore_classes=("human", "person"),
        )
        pose, ppe = FakeYOLO.instances
        return det, pose, ppe

    return factory


IMAGE = np.zeros((100, 100, 3), np.uint8)
KPS = [[[float(i), float(i), 0.9] for i in range(17)]]


def test_persons_and_ppe_are_parsed(make_detector):
    det, pose, ppe = make_detector()
    pose.result = FakeResult([[10, 20, 50, 90]], ids=[7], keypoints=KPS)
    ppe.result = FakeResult([[15, 15, 30, 30], [10, 20, 50, 90], [12, 40, 48, 70]], cls=[0, 1, 2])

    persons, items = det.detect(IMAGE)
    [p] = persons
    assert p.track_id == 7 and p.box == (10.0, 20.0, 50.0, 90.0)
    assert len(p.keypoints) == 17 and p.keypoints[3] == (3.0, 3.0, pytest.approx(0.9))
    assert [i.class_name for i in items] == ["helmet", "vest"]  # 'human' отфильтрован


def test_no_track_ids_yet_gives_no_persons(make_detector):
    det, pose, ppe = make_detector()
    pose.result = FakeResult([[10, 20, 50, 90]], ids=None, keypoints=KPS)
    ppe.result = FakeResult([])
    persons, items = det.detect(IMAGE)
    assert persons == [] and items == []


def test_tracking_is_persistent(make_detector):
    det, pose, ppe = make_detector()
    pose.result, ppe.result = FakeResult([]), FakeResult([])
    det.detect(IMAGE)
    kind, kwargs = pose.calls[0]
    assert kind == "track" and kwargs["persist"] is True and kwargs["tracker"] == "bytetrack.yaml"
    assert ppe.calls[0][0] == "predict"


@pytest.mark.parametrize("ppe_name,half,expect_half", [("ppe.pt", True, True), ("ppe.onnx", True, False), ("ppe.pt", False, False)])
def test_half_passed_only_for_pytorch_weights(make_detector, ppe_name, half, expect_half):
    det, pose, ppe = make_detector(ppe_name, half=half)
    pose.result, ppe.result = FakeResult([]), FakeResult([])
    det.detect(IMAGE)
    assert ("half" in ppe.calls[0][1]) is expect_half


def test_ppe_classes(make_detector):
    det, _, _ = make_detector()
    assert det.ppe_classes == {"helmet", "human", "vest"}


def test_missing_weights_is_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    with pytest.raises(FileNotFoundError, match="не найден"):
        YoloDetector(tmp_path / "a.pt", tmp_path / "b.pt", device="cpu", imgsz=640, half=False,
                     person_conf=0.5, ppe_conf=0.35, iou=0.5, tracker="bytetrack.yaml", ignore_classes=())


# ------------------------------------------------------------------ реальные модели


@pytest.mark.slow
def test_real_models_find_people_with_pose():
    require_real_assets("yolov8n-pose.pt", "PPE_M_YOLO.pt", "test_neg1.mp4")
    import cv2

    cap = cv2.VideoCapture(str(VIDEOS / "test_neg1.mp4"))
    ok, image = cap.read()
    cap.release()
    assert ok
    det = YoloDetector(
        WEIGHTS / "yolov8n-pose.pt", WEIGHTS / "PPE_M_YOLO.pt", device="cpu", imgsz=640, half=False,
        person_conf=0.5, ppe_conf=0.35, iou=0.5, tracker="bytetrack.yaml", ignore_classes=("human", "person"),
    )
    persons = []
    for _ in range(3):  # трекеру нужно пару кадров, чтобы выдать id
        persons, items = det.detect(image)
    assert len(persons) == 3
    assert all(len(p.keypoints) == 17 for p in persons)
    assert {"helmet", "vest", "boots", "gloves"} >= {i.class_name for i in items}
    assert det.ppe_classes == {"boots", "gloves", "helmet", "human", "vest"}
