import json
import time
from datetime import datetime

import pytest
import requests

from ppe_monitor.config import ReceiverConfig, VisDashConfig
from ppe_monitor.domain.models import CloseReason, EpisodeEvent, EpisodeKind, Rule
from ppe_monitor.sinks.base import EvidenceFile, IncidentPackage
from ppe_monitor.sinks.files import FileSink
from ppe_monitor.sinks.outbox import Outbox, SendResult
from ppe_monitor.sinks.receiver import ReceiverSink
from ppe_monitor.sinks.visdash import VisDashSink, cam_id_from


def make_package(kind=EpisodeKind.OPENED, episode_id="ep1", local_raw=False, clip=False):
    closed = kind is EpisodeKind.CLOSED
    event = EpisodeEvent(
        kind=kind, episode_id=episode_id, camera_id="cam_7", rule=Rule.PPE, subject="helmet",
        track_id=3, track_ids=(3,), started_ts=0.0, event_ts=8.0 if closed else 3.0,
        box=(10.4, 20.6, 110.0, 300.0), confidence=0.87654, violating_sec=3.0,
        ended_ts=3.0 if closed else None, close_reason=CloseReason.RESOLVED if closed else None,
    )
    files = [
        EvidenceFile("frame_start.jpg", "frame_start", b"START", "image/jpeg"),
        EvidenceFile("raw_start.jpg", "raw_start", b"RAW", "image/jpeg", local_only=local_raw),
    ]
    if clip:
        files.append(EvidenceFile("clip.mp4", "clip", b"CLIP" * 100, "video/mp4"))
    t = datetime(2026, 10, 20, 12, 0, 0)
    return IncidentPackage(
        event=event, files=tuple(files), source="test.mp4", frame_index=42, started_at=t,
        event_at=t, ended_at=t if closed else None, present_classes=("vest",),
    )


def wait_until(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code
        self.text = ""


class FakeSession:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)  # HTTP-код или исключение; последний повторяется
        self.calls = []

    def post(self, url, **kwargs):
        files = kwargs.get("files")
        if isinstance(files, list):  # файлы читаются в момент отправки
            kwargs = kwargs | {"files": [(f, (n, fh.read(), ct)) for f, (n, fh, ct) in files]}
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)


# ------------------------------------------------------------------ manifest


def test_manifest_lists_only_uploaded_files():
    m = make_package(local_raw=True).manifest()
    assert m["schema"] == "ppe-incident/1"
    assert m["status"] == "opened"
    assert [e["name"] for e in m["evidence"]] == ["frame_start.jpg"]
    assert m["person_box"] == [10, 21, 110, 300]


# ------------------------------------------------------------------ FileSink


def test_file_sink_writes_episode_folder_and_journal(tmp_path):
    sink = FileSink(tmp_path / "inc", tmp_path / "logs" / "journal.jsonl")  # папок ещё нет
    sink.publish(make_package())
    sink.publish(make_package(EpisodeKind.CLOSED, clip=True))
    sink.close()

    folder = tmp_path / "inc" / "2026-10-20" / "ep1"
    assert sorted(p.name for p in folder.iterdir()) == [
        "clip.mp4", "closed.json", "frame_start.jpg", "opened.json", "raw_start.jpg",
    ]
    records = [json.loads(line) for line in (tmp_path / "logs" / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["status"] for r in records] == ["opened", "closed"]
    assert records[1]["duration_sec"] == 3.0


# ------------------------------------------------------------------ Outbox


def test_outbox_priority_lane_first(tmp_path):
    box = Outbox(tmp_path, ["alerts", "packages"])
    box.put("packages", "p1", {}, {"a": b"1"})
    box.put("alerts", "a1", {}, {})
    assert box.next().path.name == "a1"
    box.remove(box.next())
    assert box.next().path.name == "p1"


def test_outbox_put_is_idempotent(tmp_path):
    box = Outbox(tmp_path, ["alerts"])
    box.put("alerts", "x", {"v": 1}, {})
    box.put("alerts", "x", {"v": 2}, {})
    assert box.count() == 1
    assert box.next().meta == {"v": 1}


# ------------------------------------------------------------------ ReceiverSink


@pytest.fixture
def receiver_cfg(tmp_path):
    return ReceiverConfig(url="http://rx/api", outbox_dir=tmp_path / "rx", retry_max_delay_sec=0.05)


def test_receiver_sends_manifest_and_files(receiver_cfg):
    session = FakeSession(201)
    sink = ReceiverSink(receiver_cfg, session=session, idle_poll_sec=0.05)
    sink.publish(make_package(local_raw=True))
    assert wait_until(lambda: session.calls)
    sink.close()

    [call] = session.calls
    manifest = json.loads(call["data"]["manifest"])
    assert manifest["episode_id"] == "ep1"
    assert [name for _, (name, _, _) in call["files"]] == ["frame_start.jpg"]  # raw — только локально
    assert call["headers"]["Idempotency-Key"].startswith("ep1:opened:")


def test_receiver_alerts_go_before_heavy_packages(receiver_cfg):
    # Сервер недоступен: копим closed-пакет с клипом, потом приходит новая тревога.
    session = FakeSession(requests.ConnectionError("down"))
    sink = ReceiverSink(receiver_cfg, session=session, idle_poll_sec=0.05)
    sink.publish(make_package(EpisodeKind.CLOSED, episode_id="old", clip=True))
    sink.publish(make_package(EpisodeKind.OPENED, episode_id="new"))
    sink.close()

    session = FakeSession(201)
    sink = ReceiverSink(receiver_cfg, session=session, idle_poll_sec=0.05)
    assert wait_until(lambda: len(session.calls) == 2)
    sink.close()
    order = [json.loads(c["data"]["manifest"])["episode_id"] for c in session.calls]
    assert order == ["new", "old"]


def test_receiver_retries_until_server_is_back(receiver_cfg):
    session = FakeSession(requests.ConnectionError("down"), 503, 201)
    sink = ReceiverSink(receiver_cfg, session=session, idle_poll_sec=0.05)
    sink.publish(make_package())
    assert wait_until(lambda: len(session.calls) == 3)
    assert wait_until(lambda: not list((receiver_cfg.outbox_dir / "alerts").iterdir()))
    sink.close()


def test_receiver_rejected_package_goes_to_failed(receiver_cfg):
    sink = ReceiverSink(receiver_cfg, session=FakeSession(400), idle_poll_sec=0.05)
    sink.publish(make_package())
    assert wait_until(lambda: list((receiver_cfg.outbox_dir / "failed").iterdir()))
    sink.close()


# ------------------------------------------------------------------ VisDashSink


@pytest.fixture
def visdash_cfg(tmp_path):
    return VisDashConfig(
        enabled=True, url="http://dash/api", room_id=5, outbox_dir=tmp_path / "vd",
        retry_max_delay_sec=0.05, incident_types={"helmet": 1, "zone": 4, "default": 0},
    )


def test_visdash_sends_only_opened(visdash_cfg):
    session = FakeSession(200)
    sink = VisDashSink(visdash_cfg, session=session, idle_poll_sec=0.05)
    sink.publish(make_package(EpisodeKind.OPENED))
    sink.publish(make_package(EpisodeKind.CLOSED))
    assert wait_until(lambda: session.calls)
    time.sleep(0.2)
    sink.close()

    [call] = session.calls
    assert json.loads(call["data"]["incident"]) == {"CamId": 7, "RoomId": 5, "Type": 1, "Priority": 1}
    assert call["files"]["image"][1] == b"START"
    assert call["headers"]["Idempotency-Key"] == "ep1"


def test_visdash_incident_type_mapping(visdash_cfg):
    sink = VisDashSink(visdash_cfg, session=FakeSession(200))
    try:
        assert sink.incident_type("helmet", Rule.PPE) == 1
        assert sink.incident_type("zone:crane", Rule.ZONE) == 4
        assert sink.incident_type("vest", Rule.PPE) == 0
    finally:
        sink.close()


def test_cam_id_from():
    assert cam_id_from("cam_25") == 25
    assert cam_id_from("cam_entrance") == 0


def test_send_result_values():
    assert {r.value for r in SendResult} == {"sent", "retry", "rejected"}


# ------------------------------------------------------------------ остановка


class SlowSession(FakeSession):
    def post(self, url, **kwargs):
        time.sleep(0.2)
        return super().post(url, **kwargs)


def test_close_drains_outbox_when_server_is_up(receiver_cfg):
    """При выходе (например, конец видео) последние пакеты досылаются, а не ждут следующего запуска."""
    session = SlowSession(201)
    sink = ReceiverSink(receiver_cfg, session=session, idle_poll_sec=0.05)
    for i in range(4):
        sink.publish(make_package(EpisodeKind.CLOSED, episode_id=f"ep{i}"))
    sink.close()
    assert len(session.calls) == 4
    assert not list((receiver_cfg.outbox_dir / "packages").iterdir())


def test_close_does_not_hang_when_server_is_down(receiver_cfg):
    sink = ReceiverSink(receiver_cfg, session=FakeSession(requests.ConnectionError("down")), idle_poll_sec=0.05)
    sink.publish(make_package())
    started = time.monotonic()
    sink.close()
    assert time.monotonic() - started < 3
    assert list((receiver_cfg.outbox_dir / "alerts").iterdir())  # пакет сохранён до следующего запуска
