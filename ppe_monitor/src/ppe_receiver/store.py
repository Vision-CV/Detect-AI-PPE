"""Хранилище приёмника: файлы на диске + индекс в SQLite.

Эпизод собирается из нескольких пакетов (opened → updated… → closed).
Пакеты могут прийти не по порядку (сеть, повторы), поэтому статус эпизода
никогда не «откатывается»: closed не перезаписывается поздним opened.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

STATUS_RANK = {"opened": 0, "updated": 1, "closed": 2}
SAFE_NAME = re.compile(r"^[\w.-]{1,100}$")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id     TEXT PRIMARY KEY,
    camera_id      TEXT NOT NULL,
    violation_type TEXT NOT NULL,
    rule           TEXT NOT NULL,
    status         TEXT NOT NULL,
    track_ids      TEXT NOT NULL,
    started_at     TEXT,
    ended_at       TEXT,
    duration_sec   REAL,
    close_reason   TEXT,
    updated_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS packages (
    idempotency_key TEXT PRIMARY KEY,
    episode_id      TEXT NOT NULL,
    status          TEXT NOT NULL,
    received_at     TEXT NOT NULL,
    manifest        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    episode_id   TEXT NOT NULL,
    name         TEXT NOT NULL,
    kind         TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size         INTEGER NOT NULL,
    sha256       TEXT NOT NULL,
    PRIMARY KEY (episode_id, name)
);
CREATE INDEX IF NOT EXISTS idx_episodes_started ON episodes(started_at DESC);
CREATE TABLE IF NOT EXISTS metrics_incidents (
    guid TEXT PRIMARY KEY, camId INTEGER NOT NULL, roomId INTEGER NOT NULL,
    status INTEGER NOT NULL, type INTEGER NOT NULL, priority INTEGER NOT NULL,
    date TEXT NOT NULL, artifact BLOB, creation_hash BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_date ON metrics_incidents(date DESC);
"""


class Store:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.files_dir = data_dir / "incidents"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(data_dir / "receiver.db", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA_SQL)
        self._lock = threading.Lock()

    def has_package(self, key: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM packages WHERE idempotency_key = ?", (key,)).fetchone() is not None

    def save(self, key: str, manifest: dict[str, Any], files: dict[str, bytes]) -> None:
        episode_id = manifest["episode_id"]
        folder = self.files_dir / episode_id
        folder.mkdir(exist_ok=True)
        for name, data in files.items():
            (folder / name).write_bytes(data)

        now = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO packages VALUES (?, ?, ?, ?, ?)",
                (key, episode_id, manifest["status"], now, json.dumps(manifest, ensure_ascii=False)),
            )
            for ev in manifest["evidence"]:
                self._db.execute(
                    "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?, ?)",
                    (episode_id, ev["name"], ev["kind"], ev["content_type"], ev["size"], ev["sha256"]),
                )
            row = self._db.execute("SELECT status FROM episodes WHERE episode_id = ?", (episode_id,)).fetchone()
            values = (
                manifest["camera_id"], manifest["violation_type"], manifest["rule"], manifest["status"],
                json.dumps(manifest["track_ids"]), manifest["started_at"], manifest["ended_at"],
                manifest["duration_sec"], manifest["close_reason"], now,
            )
            if row is None:
                self._db.execute("INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (episode_id, *values))
            elif STATUS_RANK[manifest["status"]] >= STATUS_RANK[row["status"]]:
                self._db.execute(
                    """UPDATE episodes SET camera_id=?, violation_type=?, rule=?, status=?, track_ids=?,
                       started_at=?, ended_at=?, duration_sec=?, close_reason=?, updated_at=?
                       WHERE episode_id=?""",
                    (*values, episode_id),
                )

    def list_episodes(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM episodes ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._episode_dict(r) for r in rows]

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)).fetchone()
            if row is None:
                return None
            result = self._episode_dict(row)
            result["packages"] = [
                {"status": p["status"], "received_at": p["received_at"], "manifest": json.loads(p["manifest"])}
                for p in self._db.execute(
                    "SELECT * FROM packages WHERE episode_id = ? ORDER BY received_at", (episode_id,)
                )
            ]
            return result

    def file_path(self, episode_id: str, name: str) -> Path | None:
        if not SAFE_NAME.match(name) or not SAFE_NAME.match(episode_id):
            return None
        path = self.files_dir / episode_id / name
        return path if path.is_file() else None

    def _episode_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["track_ids"] = json.loads(d["track_ids"])
        d["files"] = [
            dict(f) for f in self._db.execute(
                "SELECT name, kind, content_type, size, sha256 FROM files WHERE episode_id = ? ORDER BY name",
                (row["episode_id"],),
            )
        ]
        return d

    def close(self) -> None:
        self._db.close()

    def create_incident(self, fields: dict, image: bytes, *, generated_date: bool) -> bool:
        original = {**fields, "date": None if generated_date else fields["date"]}
        digest = hashlib.sha256(json.dumps(original, sort_keys=True).encode())
        digest.update(image)
        fingerprint = digest.digest()
        with self._lock, self._db:
            inserted = self._db.execute(
                """INSERT INTO metrics_incidents (guid, camId, roomId, status, type, priority, date, artifact, creation_hash)
                   VALUES (:guid, :camId, :roomId, :status, :type, :priority, :date, :artifact, :creation_hash)
                   ON CONFLICT(guid) DO NOTHING""",
                {**fields, "artifact": image or None, "creation_hash": fingerprint},
            )
            if inserted.rowcount:
                return True
            existing = self._db.execute(
                "SELECT creation_hash FROM metrics_incidents WHERE guid=?", (fields["guid"],)
            ).fetchone()
            return existing[0] == fingerprint

    def incidents(self, guid: str | None = None) -> list[dict]:
        query = "SELECT guid, camId, roomId, status, type, priority, date FROM metrics_incidents"
        with self._lock:
            if guid is not None:
                rows = self._db.execute(query + " WHERE guid=?", (guid,))
            else:
                rows = self._db.execute(query + " ORDER BY date DESC, rowid DESC")
            return [dict(row) for row in rows]

    def artifact(self, guid: str) -> bytes | None:
        with self._lock:
            row = self._db.execute("SELECT artifact FROM metrics_incidents WHERE guid=?", (guid,)).fetchone()
            return row[0] if row else None

    def update_incident(self, fields: dict) -> bool:
        with self._lock, self._db:
            return self._db.execute(
                """UPDATE metrics_incidents SET camId=:camId, roomId=:roomId, status=:status,
                   type=:type, priority=:priority WHERE guid=:guid""", fields,
            ).rowcount == 1

    def delete_incident(self, guid: str) -> bool:
        with self._lock, self._db:
            return self._db.execute("DELETE FROM metrics_incidents WHERE guid=?", (guid,)).rowcount == 1
