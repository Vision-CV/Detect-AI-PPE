"""Локальная копия пакетов: папка на эпизод + строка в JSONL-журнале на событие.

    data/incidents/2026-10-20/<episode_id>/
        frame_start.jpg  raw_start.jpg  frame_best.jpg  frame_end.jpg  clip.mp4
        opened.json  closed.json        ← манифесты (со всеми файлами, включая локальные)
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ppe_monitor.sinks.base import IncidentPackage

logger = logging.getLogger(__name__)


class FileSink:
    def __init__(self, incidents_dir: Path, journal_file: Path) -> None:
        self._dir = incidents_dir
        self._journal = journal_file
        self._dir.mkdir(parents=True, exist_ok=True)
        self._journal.parent.mkdir(parents=True, exist_ok=True)
        # Один рабочий поток: записи журнала идут строго по порядку, без блокировок.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="FileSink")

    def publish(self, package: IncidentPackage) -> None:
        self._executor.submit(self._write, package)

    def _write(self, package: IncidentPackage) -> None:
        try:
            e = package.event
            folder = self._dir / package.started_at.strftime("%Y-%m-%d") / e.episode_id
            folder.mkdir(parents=True, exist_ok=True)
            for f in package.files:
                (folder / f.name).write_bytes(f.data)
            manifest = package.manifest(include_local=True)
            (folder / f"{package.status}.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            record = {k: v for k, v in manifest.items() if k != "evidence"} | {
                "folder": str(folder),
                "files": [f.name for f in package.files],
            }
            with self._journal.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("Сохранён пакет %s %s: %s", e.violation_type, package.status, folder.name[:8])
        except Exception:
            logger.exception("Не удалось сохранить пакет инцидента")

    def close(self) -> None:
        self._executor.shutdown(wait=True)
