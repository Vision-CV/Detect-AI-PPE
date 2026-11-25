import io
import json
import time
from uuid import UUID, uuid4

import jwt
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ppe_receiver.app import create_app


BASE = "/api/Metrics"
FIELDS = {"camId": 12, "roomId": 34, "status": 0, "type": 0, "priority": 1}
SECRET = "a-local-test-secret-of-at-least-32-bytes"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PPE_METRICS_API_KEY", "test")
    monkeypatch.setenv("PPE_METRICS_JWT_SECRET", SECRET)
    monkeypatch.setenv("PPE_METRICS_JWT_AUDIENCE", "metrics")
    monkeypatch.delenv("PPE_METRICS_JWT_ISSUER", raising=False)
    with TestClient(create_app(tmp_path), headers={"X-API-Key": "test"}) as api:
        yield api


def post(client, fields, image):
    return client.post(BASE + "/send-incident", data={"incident": json.dumps(fields)},
                       files={"image": ("image.jpg", image, "image/jpeg")})


def test_incident_lifecycle(client):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, "JPEG")
    image = buffer.getvalue()
    fields = {**FIELDS, "guid": str(uuid4()), "date": "2020-04-01T10:00:00+04:00"}
    created = post(client, fields, image)
    assert created.status_code == 200 and created.text == fields["guid"]
    path = BASE + "/incidents/" + created.text
    first = client.get(path).json()
    assert first == {**fields, "date": "2020-04-01T06:00:00Z"}
    updated = {**first, "status": 1, "date": "2025-01-01T00:00:00Z"}
    assert client.patch(BASE + "/patch-incident", json=updated).status_code == 200
    assert post(client, fields, image).status_code == 200
    assert post(client, {**fields, "priority": 2}, image).status_code == 409
    assert client.get(path).json() == {**first, "status": 1}
    artifact = client.get(path + "/artifact")
    assert artifact.headers["content-type"] == "image/jpeg" and artifact.content == image
    other = post(client, FIELDS, b"")
    assert other.status_code == 200 and UUID(other.text)
    assert [row["guid"] for row in client.get(BASE + "/incidents").json()] == [other.text, created.text]
    assert client.get(BASE + "/incidents/" + other.text + "/artifact").status_code == 404
    params = {"guid": first["guid"]}
    assert client.delete(BASE + "/delete-incident", params=params).status_code == 200
    assert client.get(path).status_code == client.get(path + "/artifact").status_code == 404
    assert client.patch(BASE + "/patch-incident", json=updated).status_code == 400
    assert client.delete(BASE + "/delete-incident", params=params).status_code == 400


def test_authentication_and_roles(client):
    client.headers.pop("X-API-Key")
    listed = BASE + "/incidents"
    assert client.get(listed).status_code == 401
    assert client.get(listed, headers={"X-API-Key": "wrong"}).status_code == 401
    claims = {"role": "Default", "aud": "metrics", "exp": int(time.time()) + 300}
    for changes, secret, expected in [
        ({}, SECRET, 200), ({"exp": 1}, SECRET, 401), ({"aud": "other"}, SECRET, 401),
        ({}, "an-untrusted-test-key-of-sufficient-length", 401),
    ]:
        token = jwt.encode({**claims, **changes}, secret, algorithm="HS256")
        assert client.get(listed, headers={"Authorization": f"Bearer {token}"}).status_code == expected
    client.headers["Authorization"] = "Bearer " + jwt.encode(claims, SECRET, algorithm="HS256")
    assert post(client, FIELDS, b"").status_code == 403
    guid = str(uuid4())
    assert client.patch(BASE + "/patch-incident", json={**FIELDS, "guid": guid}).status_code == 403
    assert client.delete(BASE + "/delete-incident", params={"guid": guid}).status_code == 403
    client.headers["Authorization"] = "Bearer " + jwt.encode({**claims, "role": "SA"}, SECRET, algorithm="HS256")
    assert post(client, FIELDS, b"").status_code == 200


def test_rejects_invalid_input(client):
    assert post(client, {**FIELDS, "type": 3}, b"").status_code == 400
    assert post(client, FIELDS, b"not a jpeg").status_code == 400
    assert client.patch(BASE + "/patch-incident", json={"guid": str(uuid4()), "status": 2}).status_code == 422
    assert client.get(BASE + "/incidents/not-a-guid").status_code == 422
