"""HTTP API приёмника.

POST /api/v1/incidents                     — принять пакет (manifest + files)
GET  /api/v1/incidents                     — список эпизодов
GET  /api/v1/incidents/{id}                — эпизод со всеми пакетами
GET  /api/v1/incidents/{id}/files/{name}   — файл-доказательство
GET  /                                      — простая страница для просмотра
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ppe_receiver.store import SAFE_NAME, STATUS_RANK, Store
from ppe_receiver.metrics import metrics_router

logger = logging.getLogger("ppe_receiver")

SCHEMA = "ppe-incident/1"
REQUIRED = (
    "schema", "episode_id", "status", "camera_id", "rule", "violation_type",
    "track_ids", "started_at", "ended_at", "duration_sec", "close_reason", "evidence",
)
MAX_FILE_BYTES = 100 * 1024 * 1024


def create_app(data_dir: Path) -> FastAPI:
    store = Store(data_dir)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        store.close()

    app = FastAPI(title="PPE Incident Receiver", version="1.0", lifespan=lifespan)
    app.state.store = store
    app.include_router(metrics_router(store))

    @app.post("/api/v1/incidents", status_code=201)
    async def receive(
        manifest: Annotated[str, Form()],
        files: Annotated[list[UploadFile], File()] = [],  # noqa: B006 — так FastAPI задаёт «необязательно»
        idempotency_key: Annotated[str | None, Header()] = None,
    ):
        data = _parse_manifest(manifest)
        key = idempotency_key or f"{data['episode_id']}:{data['status']}:{data.get('event_at')}"
        if store.has_package(key):
            return JSONResponse({"episode_id": data["episode_id"], "duplicate": True}, status_code=200)

        received = {}
        for upload in files:
            content = await upload.read(MAX_FILE_BYTES + 1)
            if len(content) > MAX_FILE_BYTES:
                raise HTTPException(413, f"Файл {upload.filename} больше {MAX_FILE_BYTES} байт")
            received[upload.filename] = content
        _verify_files(data["evidence"], received)

        store.save(key, data, received)
        logger.info("Принят пакет %s %s (%s), файлов: %d",
                    data["violation_type"], data["status"], data["episode_id"][:8], len(received))
        return {"episode_id": data["episode_id"], "status": data["status"], "duplicate": False}

    @app.get("/api/v1/incidents")
    def list_incidents(limit: int = 100):
        return store.list_episodes(min(max(limit, 1), 1000))

    @app.get("/api/v1/incidents/{episode_id}")
    def get_incident(episode_id: str):
        episode = store.get_episode(episode_id)
        if episode is None:
            raise HTTPException(404, "Эпизод не найден")
        return episode

    @app.get("/api/v1/incidents/{episode_id}/files/{name}")
    def get_file(episode_id: str, name: str):
        path = store.file_path(episode_id, name)
        if path is None:
            raise HTTPException(404, "Файл не найден")
        return FileResponse(path)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _render_index(store.list_episodes(200))

    return app


def _parse_manifest(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise HTTPException(400, f"manifest: некорректный JSON: {e}") from e
    if not isinstance(data, dict):
        raise HTTPException(400, "manifest должен быть объектом")
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise HTTPException(400, f"manifest: нет полей {missing}")
    if data["schema"] != SCHEMA:
        raise HTTPException(400, f"manifest: неподдерживаемая схема {data['schema']!r}, ожидается {SCHEMA}")
    if data["status"] not in STATUS_RANK:
        raise HTTPException(400, f"manifest: неизвестный статус {data['status']!r}")
    if not SAFE_NAME.match(str(data["episode_id"])):
        raise HTTPException(400, "manifest: недопустимый episode_id")
    return data


def _verify_files(evidence: list[dict], received: dict[str, bytes]) -> None:
    """Каждый файл из manifest должен прийти, совпасть по SHA-256; лишних — нет."""
    declared = {e["name"]: e for e in evidence}
    for name in received:
        if name not in declared or not SAFE_NAME.match(name):
            raise HTTPException(400, f"Файл {name!r} не объявлен в manifest.evidence")
    for name, meta in declared.items():
        if name not in received:
            raise HTTPException(400, f"Не получен файл {name!r} из manifest.evidence")
        digest = hashlib.sha256(received[name]).hexdigest()
        if digest != meta["sha256"]:
            raise HTTPException(400, f"Файл {name!r}: SHA-256 не совпадает (повреждён при передаче?)")


def _render_index(episodes: list[dict]) -> str:
    rows = []
    for ep in episodes:
        eid = html.escape(ep["episode_id"])
        names = {f["name"] for f in ep["files"]}
        thumb = next((n for n in ("frame_best.jpg", "frame_start.jpg") if n in names), None)
        img = f'<img src="/api/v1/incidents/{eid}/files/{thumb}" loading="lazy">' if thumb else ""
        links = " ".join(
            f'<a href="/api/v1/incidents/{eid}/files/{html.escape(f["name"])}">{html.escape(f["kind"])}</a>'
            for f in ep["files"]
        )
        duration = "" if ep["duration_sec"] is None else f'{ep["duration_sec"]:.1f} с'
        rows.append(
            f'<tr><td>{img}</td><td><b>{html.escape(ep["violation_type"])}</b><br>{html.escape(ep["camera_id"])}'
            f'<br>ID {html.escape(", ".join(map(str, ep["track_ids"])))}</td>'
            f'<td class="{html.escape(ep["status"])}">{html.escape(ep["status"])}'
            f'<br>{html.escape(ep["close_reason"] or "")}</td>'
            f'<td>{html.escape(ep["started_at"] or "")}<br>{duration}</td>'
            f'<td>{links}<br><a href="/api/v1/incidents/{eid}">json</a></td></tr>'
        )
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="refresh" content="10">
<title>Инциденты СИЗ</title><style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#f6f7f9;color:#1d2330}}
table{{border-collapse:collapse;width:100%;background:#fff}}
td,th{{border-bottom:1px solid #e3e6eb;padding:8px;text-align:left;vertical-align:top;font-size:14px}}
img{{width:220px;border-radius:4px}} .opened,.updated{{color:#c62828;font-weight:600}} .closed{{color:#2e7d32}}
</style></head><body><h1>Инциденты СИЗ</h1><p>Эпизодов: {len(episodes)} · обновление каждые 10 с</p>
<table><tr><th>Кадр</th><th>Нарушение</th><th>Статус</th><th>Начало</th><th>Доказательства</th></tr>
{''.join(rows) or '<tr><td colspan="5">Пока нет инцидентов</td></tr>'}</table></body></html>"""
