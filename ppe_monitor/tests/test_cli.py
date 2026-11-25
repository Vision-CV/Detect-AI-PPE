"""Точка входа и сборка приложения: аргументы, коды выхода, полный прогон с фейковым детектором."""

import json
import logging

import pytest

from ppe_monitor import cli
from ppe_monitor.config import load_config
from ppe_monitor.domain.models import Person, PPEItem

from conftest import write_video


@pytest.fixture(autouse=True)
def reset_logging():
    """cli настраивает корневой логгер с файлом — закрываем, чтобы Windows отпустила tmp."""
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)


class FakeDetector:
    """Подменяет YoloDetector: один человек в жилете, без каски."""

    classes = {"helmet", "vest", "human"}

    def __init__(self, pose, ppe, **kwargs):
        self.kwargs = kwargs

    @property
    def ppe_classes(self):
        return self.classes

    def detect(self, image):
        kps = tuple((60.0, 10.0 + i * 6, 0.9) for i in range(17))
        worker = Person(1, (40.0, 8.0, 80.0, 112.0), 0.9, kps)
        return [worker], [PPEItem("vest", (45.0, 30.0, 75.0, 65.0), 0.9)]


def write_config(tmp_path, video, **extra):
    cfg = {
        "camera_id": "cam_test",
        "source": {"uri": str(video), "process_every_n_frames": 1},
        "hardware": {"device": "cpu"},
        "storage": {"incidents_dir": "out/incidents", "journal_file": "out/journal.jsonl"},
        "receiver": {"enabled": False},
        "visdash": {"enabled": False},
        "display": {"enabled": False},
        "logging": {"file": "out/app.log"},
        "evidence": {"clip_pre_sec": 1, "clip_post_sec": 1},
    } | extra
    path = tmp_path / "cfg.yaml"
    path.write_text(json.dumps(cfg), encoding="utf-8")  # JSON — подмножество YAML
    return path


# ------------------------------------------------------------------ аргументы


def test_cli_flags_become_config_overrides(tmp_path):
    args = cli.parse_args(["--source", "v.mp4", "--camera-id", "cam_9", "--resize", "640",
                           "--device", "cuda", "--no-display", "--no-api", "--zones", "z.yaml"])
    o = cli.overrides_from(args)
    assert o["source.uri"] == "v.mp4" and o["camera_id"] == "cam_9"
    assert o["source.resize_width"] == 640 and o["hardware.device"] == "cuda"
    assert o["display.enabled"] is False
    assert o["receiver.enabled"] is False and o["visdash.enabled"] is False
    assert o["rules.zones_file"].endswith("z.yaml")


def test_no_flags_no_overrides():
    assert cli.overrides_from(cli.parse_args([])) == {}


def test_overrides_apply_to_project_config():
    config = load_config(None, cli.overrides_from(cli.parse_args(["--no-api", "--source", "x.mp4"])))
    assert config.source.uri == "x.mp4" and not config.receiver.enabled and not config.visdash.enabled


# ------------------------------------------------------------------ коды выхода


def test_bad_config_exits_2(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: {ppe_conff: 1}", encoding="utf-8")
    assert cli.main(["--config", str(bad)]) == 2
    assert "ppe_conff" in capsys.readouterr().err


def test_missing_video_exits_1(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("ppe_monitor.pipeline.detector.YoloDetector", FakeDetector)
    cfg = write_config(tmp_path, tmp_path / "nope.mp4")
    assert cli.main(["--config", str(cfg)]) == 1


def test_model_without_required_class_exits_1(tmp_path, monkeypatch, synthetic_video):
    class NoHelmet(FakeDetector):
        classes = {"vest", "human"}

    monkeypatch.setattr("ppe_monitor.pipeline.detector.YoloDetector", NoHelmet)
    assert cli.main(["--config", str(write_config(tmp_path, synthetic_video))]) == 1
    assert "helmet" in (tmp_path / "out" / "app.log").read_text(encoding="utf-8")


# ------------------------------------------------------------------ полный прогон


def test_full_run_produces_incident_package(tmp_path, monkeypatch):
    monkeypatch.setattr("ppe_monitor.pipeline.detector.YoloDetector", FakeDetector)
    video = write_video(tmp_path / "v.avi", 60)  # 6 с при 10 к/с
    assert cli.main(["--config", str(write_config(tmp_path, video)), "--max-frames", "60"]) == 0

    records = [json.loads(line) for line in (tmp_path / "out" / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["status"], r["violation_type"]) for r in records] == [("opened", "helmet"), ("closed", "helmet")]
    assert records[0]["camera_id"] == "cam_test"
    assert records[1]["close_reason"] == "stream_ended"
    folder = tmp_path / "out" / "incidents"
    [episode] = [p for p in folder.glob("*/*") if p.is_dir()]
    assert {"frame_start.jpg", "raw_start.jpg", "clip.mp4", "opened.json", "closed.json"} <= {p.name for p in episode.iterdir()}
    log = (tmp_path / "out" / "app.log").read_text(encoding="utf-8")
    assert "нарушений открыто: 1, закрыто: 1" in log
