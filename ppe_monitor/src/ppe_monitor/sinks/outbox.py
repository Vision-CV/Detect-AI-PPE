"""Outbox на диске с приоритетными очередями и фоновой отправкой.

Каждый элемент — папка (meta.json + файлы). Сначала пишется во временную
папку и атомарно переносится в очередь: отправитель видит только целые
элементы. Удаляется только после успешной отправки, поэтому переживает
недоступность сервера и перезапуск программы.

Очереди (lanes) идут в порядке приоритета: перед КАЖДОЙ отправкой берётся
первый элемент самой приоритетной непустой очереди. Тревога, пришедшая во
время отправки тяжёлого клипа, уйдёт следующей, а не после всех клипов.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

META = "meta.json"
FAILED = "failed"
TMP = ".tmp"


class SendResult(Enum):
    SENT = "sent"  # принято — удалить
    RETRY = "retry"  # сервер недоступен — повторить позже
    REJECTED = "rejected"  # сервер отклонил (4xx) — повтор не поможет, в failed/


@dataclass(frozen=True)
class OutboxItem:
    path: Path
    lane: str

    @property
    def meta(self) -> dict[str, Any]:
        return json.loads((self.path / META).read_text(encoding="utf-8"))

    def files(self) -> dict[str, Path]:
        return {p.name: p for p in sorted(self.path.iterdir()) if p.name != META}


class Outbox:
    def __init__(self, root: Path, lanes: Sequence[str]) -> None:
        self.root = root
        self.lanes = list(lanes)
        for lane in self.lanes:
            (root / lane).mkdir(parents=True, exist_ok=True)
        (root / FAILED).mkdir(parents=True, exist_ok=True)
        shutil.rmtree(root / TMP, ignore_errors=True)  # недописанное при прошлом падении

    def put(self, lane: str, item_id: str, meta: Mapping[str, Any], files: Mapping[str, bytes]) -> None:
        tmp = self.root / TMP / f"{item_id}-{uuid.uuid4().hex[:6]}"
        tmp.mkdir(parents=True)
        for name, data in files.items():
            (tmp / name).write_bytes(data)
        (tmp / META).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        target = self.root / lane / item_id
        if target.exists():
            shutil.rmtree(tmp)  # такой элемент уже ждёт отправки
            return
        tmp.replace(target)

    def next(self) -> OutboxItem | None:
        for lane in self.lanes:
            items = sorted((p for p in (self.root / lane).iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
            if items:
                return OutboxItem(items[0], lane)
        return None

    def count(self) -> int:
        return sum(1 for lane in self.lanes for p in (self.root / lane).iterdir() if p.is_dir())

    def remove(self, item: OutboxItem) -> None:
        shutil.rmtree(item.path, ignore_errors=True)

    def reject(self, item: OutboxItem) -> None:
        target = self.root / FAILED / item.path.name
        shutil.rmtree(target, ignore_errors=True)
        item.path.replace(target)


class OutboxSender:
    """Фоновый поток: отправляет элементы outbox по одному, с паузами при сбоях."""

    def __init__(
        self,
        outbox: Outbox,
        send: Callable[[OutboxItem], SendResult],
        *,
        name: str,
        retry_max_delay_sec: float = 60.0,
        idle_poll_sec: float = 5.0,
    ) -> None:
        self._outbox = outbox
        self._send = send
        self._name = name
        self._max_delay = retry_max_delay_sec
        self._idle = idle_poll_sec
        self._online = True
        self._stop = threading.Event()
        self._wake = threading.Event()
        pending = outbox.count()
        if pending:
            logger.info("%s: в outbox %d неотправленных элементов — отправлю", name, pending)
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def wake(self) -> None:
        self._wake.set()

    def _run(self) -> None:
        delay = 1.0
        while not self._stop.is_set():
            item = self._outbox.next()
            if item is None:
                self._wake.wait(self._idle)
                self._wake.clear()
                continue
            try:
                result = self._send(item)
            except Exception:
                logger.exception("%s: ошибка при отправке %s", self._name, item.path.name)
                result = SendResult.REJECTED

            if result is SendResult.SENT:
                self._outbox.remove(item)
                self._set_online(True)
                delay = 1.0
            elif result is SendResult.REJECTED:
                self._outbox.reject(item)
            else:
                self._set_online(False)
                self._stop.wait(delay)
                delay = min(delay * 2, self._max_delay)

    def _set_online(self, online: bool) -> None:
        if online and not self._online:
            logger.info("%s: сервер снова доступен", self._name)
        elif not online and self._online:
            logger.warning("%s: сервер недоступен, данные копятся в %s", self._name, self._outbox.root)
        self._online = online

    def close(self, drain_sec: float = 15.0, timeout: float = 5.0) -> None:
        """Перед остановкой досылает очередь (до drain_sec), если сервер доступен.

        Если сервер недоступен — не ждёт: элементы останутся в outbox и уйдут
        при следующем запуске.
        """
        deadline = time.monotonic() + drain_sec
        while self._outbox.count() and self._online and time.monotonic() < deadline:
            self._wake.set()
            time.sleep(0.05)
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout)
        left = self._outbox.count()
        if left:
            logger.info("%s: неотправленных элементов %d — отправятся при следующем запуске", self._name, left)
