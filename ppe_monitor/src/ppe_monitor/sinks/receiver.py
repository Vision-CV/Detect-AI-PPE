"""Отправка пакетов инцидентов на сервер-приёмник (src/ppe_receiver).

Контракт: POST multipart/form-data
    manifest — JSON пакета (schema "ppe-incident/1", список evidence с SHA-256)
    files    — файлы из manifest.evidence (имя файла = name)
Заголовок Idempotency-Key = "<episode_id>:<status>:<event_at>".

Две очереди: alerts (открытие, напоминание — маленькие, срочные) и
packages (закрытие с клипом — тяжёлые). Тревоги всегда уходят первыми.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from ppe_monitor.config import ReceiverConfig
from ppe_monitor.sinks.base import IncidentPackage
from ppe_monitor.sinks.outbox import Outbox, OutboxItem, OutboxSender, SendResult

logger = logging.getLogger(__name__)

ALERTS, PACKAGES = "alerts", "packages"


class ReceiverSink:
    def __init__(self, config: ReceiverConfig, session: Any | None = None, idle_poll_sec: float = 5.0) -> None:
        self._cfg = config
        self._session = session or requests.Session()
        self._outbox = Outbox(config.outbox_dir, [ALERTS, PACKAGES])
        if not config.ssl_verify:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("Проверка SSL-сертификата приёмника отключена (receiver.ssl_verify=false)")
        self._sender = OutboxSender(
            self._outbox, self._send, name="Receiver",
            retry_max_delay_sec=config.retry_max_delay_sec, idle_poll_sec=idle_poll_sec,
        )

    def publish(self, package: IncidentPackage) -> None:
        manifest = package.manifest()
        item_id = f"{manifest['episode_id']}-{manifest['status']}-{manifest['event_at'].replace(':', '')}"
        files = {f.name: f.data for f in package.upload_files()}
        meta = {
            "manifest": manifest,
            "idempotency_key": f"{manifest['episode_id']}:{manifest['status']}:{manifest['event_at']}",
            "content_types": {f.name: f.content_type for f in package.upload_files()},
        }
        self._outbox.put(ALERTS if package.is_alert else PACKAGES, item_id, meta, files)
        self._sender.wake()

    def _send(self, item: OutboxItem) -> SendResult:
        meta = item.meta
        handles = []
        try:
            files = []
            for name, path in item.files().items():
                fh = path.open("rb")
                handles.append(fh)
                files.append(("files", (name, fh, meta["content_types"].get(name, "application/octet-stream"))))
            response = self._session.post(
                self._cfg.url,
                data={"manifest": json.dumps(meta["manifest"], ensure_ascii=False)},
                files=files,
                headers={"Idempotency-Key": meta["idempotency_key"]},
                timeout=self._cfg.timeout_sec,
                verify=self._cfg.ssl_verify,
            )
        except requests.RequestException as exc:
            logger.debug("Приёмник недоступен: %s", exc)
            return SendResult.RETRY
        finally:
            for fh in handles:
                fh.close()

        status = response.status_code
        manifest = meta["manifest"]
        if 200 <= status < 300:
            logger.info("Пакет %s %s (%s) принят приёмником",
                        manifest["violation_type"], manifest["status"], manifest["episode_id"][:8])
            return SendResult.SENT
        if status == 429 or status >= 500:
            return SendResult.RETRY
        logger.error("Приёмник отклонил пакет %s (HTTP %d): %s", item.path.name, status, response.text[:300])
        return SendResult.REJECTED

    def close(self) -> None:
        self._sender.close(drain_sec=self._cfg.shutdown_drain_sec)
