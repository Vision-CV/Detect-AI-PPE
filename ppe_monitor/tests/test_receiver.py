"""Приёмник пакетов (FastAPI) + сквозная проверка: ReceiverSink → приёмник."""

import hashlib
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ppe_receiver.app import create_app  # noqa: E402


def manifest(status="opened", episode_id="ep1", files=None, **extra):
    files = files if files is not None else {"frame_start.jpg": b"START"}
    return {
        "schema": "ppe-incident/1", "episode_id": episode_id, "status": status, "camera_id": "cam1",
        "rule": "ppe", "subject": "helmet", "violation_type": "helmet", "track_ids": [3],
        "started_at": "2026-10-20T12:00:00.000", "event_at": f"2026-10-20T12:00:0{len(status)}.000",
        "ended_at": "2026-10-20T12:00:06.000" if status == "closed" else None,
        "duration_sec": 6.0 if status == "closed" else None,
        "close_reason": "resolved" if status == "closed" else None,
        "evidence": [
            {"name": n, "kind": n.split(".")[0], "content_type": "image/jpeg", "size": len(d),
             "sha256": hashlib.sha256(d).hexdigest()}
            for n, d in files.items()
        ],
    } | extra


def post(client, m, files=None, key=None):
    files = files if files is not None else {"frame_start.jpg": b"START"}
    headers = {"Idempotency-Key": key} if key else {}
    return client.post(
        "/api/v1/incidents",
        data={"manifest": json.dumps(m)},
        files=[("files", (n, d, "image/jpeg")) for n, d in files.items()],
        headers=headers,
    )


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        yield c


def test_accepts_package_and_serves_files(client):
    r = post(client, manifest())
    assert r.status_code == 201, r.text
    [ep] = client.get("/api/v1/incidents").json()
    assert ep["status"] == "opened" and ep["violation_type"] == "helmet"
    assert client.get("/api/v1/incidents/ep1/files/frame_start.jpg").content == b"START"


def test_duplicate_is_idempotent(client):
    assert post(client, manifest(), key="k1").status_code == 201
    r = post(client, manifest(), key="k1")
    assert r.status_code == 200 and r.json()["duplicate"] is True
    assert len(client.get("/api/v1/incidents/ep1").json()["packages"]) == 1


def test_closed_updates_episode(client):
    post(client, manifest("opened"))
    files = {"frame_start.jpg": b"START", "clip.mp4": b"CLIP"}
    assert post(client, manifest("closed", files=files), files=files).status_code == 201
    ep = client.get("/api/v1/incidents/ep1").json()
    assert ep["status"] == "closed" and ep["duration_sec"] == 6.0
    assert {f["name"] for f in ep["files"]} == {"frame_start.jpg", "clip.mp4"}


def test_late_opened_does_not_downgrade_closed(client):
    post(client, manifest("closed"))
    post(client, manifest("opened"))
    assert client.get("/api/v1/incidents/ep1").json()["status"] == "closed"


def test_corrupted_file_is_rejected(client):
    r = post(client, manifest(), files={"frame_start.jpg": b"BROKEN"})
    assert r.status_code == 400 and "SHA-256" in r.json()["detail"]


def test_missing_and_undeclared_files_rejected(client):
    assert post(client, manifest(), files={}).status_code == 400
    extra = {"frame_start.jpg": b"START", "evil.jpg": b"X"}
    assert post(client, manifest(), files=extra).status_code == 400


@pytest.mark.parametrize("patch", [{"schema": "other/9"}, {"status": "weird"}, {"episode_id": "../../etc"}])
def test_invalid_manifest_rejected(client, patch):
    assert post(client, manifest(**patch)).status_code == 400


def test_path_traversal_in_file_url(client):
    post(client, manifest())
    assert client.get("/api/v1/incidents/ep1/files/..%2F..%2Freceiver.db").status_code == 404


def test_index_page(client):
    post(client, manifest())
    page = client.get("/")
    assert page.status_code == 200 and "helmet" in page.text


def test_end_to_end_with_receiver_sink(client, tmp_path):
    """Настоящий ReceiverSink шлёт через TestClient (requests-совместимый интерфейс)."""
    import time

    from ppe_monitor.config import ReceiverConfig
    from ppe_monitor.sinks.receiver import ReceiverSink

    from test_sinks import make_package

    class Adapter:  # requests.Session.post → TestClient.post
        def post(self, url, data, files, headers, timeout, verify):
            return client.post("/api/v1/incidents", data=data, files=files, headers=headers)

    sink = ReceiverSink(ReceiverConfig(url="http://rx", outbox_dir=tmp_path / "out"), session=Adapter(), idle_poll_sec=0.05)
    sink.publish(make_package(local_raw=True))
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not client.get("/api/v1/incidents").json():
        time.sleep(0.05)
    sink.close()

    [ep] = client.get("/api/v1/incidents").json()
    assert ep["episode_id"] == "ep1"
    assert [f["name"] for f in ep["files"]] == ["frame_start.jpg"]
