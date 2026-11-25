"""Системные тесты: настоящие модели, настоящие ролики, настоящий HTTP-приёмник.

Медленные (~15–40 с каждый на CPU), поэтому по умолчанию не запускаются:
    py -3.11 -m pytest -m slow
"""

import json
import logging
import socket
import threading
import time

import pytest

from ppe_monitor import cli

from conftest import require_real_assets

pytestmark = pytest.mark.slow
MODELS = ("yolov8n-pose.pt", "PPE_M_YOLO.pt")


@pytest.fixture(autouse=True)
def reset_logging():
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)


def config_file(tmp_path, video, receiver_url=None):
    cfg = {
        "camera_id": "cam_1",
        "source": {"uri": str(video), "process_every_n_frames": 3},
        "hardware": {"device": "cpu"},
        "models": {"imgsz": 640},
        "storage": {"incidents_dir": str(tmp_path / "incidents"), "journal_file": str(tmp_path / "journal.jsonl")},
        "receiver": {"enabled": receiver_url is not None, "url": receiver_url or "http://x",
                     "outbox_dir": str(tmp_path / "outbox")},
        "visdash": {"enabled": False},
        "display": {"enabled": False},
        "logging": {"file": str(tmp_path / "app.log")},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def journal(tmp_path):
    path = tmp_path / "journal.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_people_without_ppe_are_reported(tmp_path):
    require_real_assets(*MODELS, "test_neg1.mp4")
    from conftest import VIDEOS

    assert cli.main(["--config", str(config_file(tmp_path, VIDEOS / "test_neg1.mp4"))]) == 0
    records = journal(tmp_path)
    opened = [r for r in records if r["status"] == "opened"]
    closed = [r for r in records if r["status"] == "closed"]
    helmet_tracks = {r["track_id"] for r in opened if r["violation_type"] == "helmet"}
    assert len(helmet_tracks) == 3  # трое без касок
    assert {r["episode_id"] for r in opened} == {r["episode_id"] for r in closed}  # каждый эпизод закрыт
    for r in opened:
        assert 2.9 <= (r["video_offsets_ms"]["event"] - r["video_offsets_ms"]["started"]) / 1000 <= 3.5
    for folder in (tmp_path / "incidents").glob("*/*"):
        assert (folder / "clip.mp4").stat().st_size > 10_000


def test_equipped_workers_are_not_reported(tmp_path):
    require_real_assets(*MODELS, "test2.mp4")
    from conftest import VIDEOS

    assert cli.main(["--config", str(config_file(tmp_path, VIDEOS / "test2.mp4"))]) == 0
    assert [r for r in journal(tmp_path) if r["status"] == "opened"] == []


@pytest.fixture
def receiver_server(tmp_path):
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    from ppe_receiver.app import create_app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = create_app(tmp_path / "receiver")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "приёмник не запустился"
    yield f"http://127.0.0.1:{port}", app.state.store
    server.should_exit = True
    thread.join(10)


def test_full_system_delivers_packages_to_receiver(tmp_path, receiver_server):
    require_real_assets(*MODELS, "test_neg1.mp4")
    from conftest import VIDEOS

    url, store = receiver_server
    cfg = config_file(tmp_path, VIDEOS / "test_neg1.mp4", receiver_url=f"{url}/api/v1/incidents")
    assert cli.main(["--config", str(cfg)]) == 0

    opened = {r["episode_id"] for r in journal(tmp_path) if r["status"] == "opened"}
    assert opened
    episodes = {e["episode_id"]: e for e in store.list_episodes()}
    assert set(episodes) == opened  # все эпизоды дошли до приёмника
    for e in episodes.values():
        assert e["status"] == "closed"  # и пакеты закрытия тоже — до выхода программы
        assert {"frame_start.jpg", "clip.mp4"} <= {f["name"] for f in e["files"]}
    assert not list((tmp_path / "outbox").glob("*/*"))  # outbox пуст (кроме failed/)
