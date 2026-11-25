"""Отправка JPEG и метаданных открытого эпизода в Metrics API."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC
from typing import Any
from uuid import UUID

import requests

from ppe_monitor.config import VisDashConfig
from ppe_monitor.sinks.base import IncidentPackage
from ppe_monitor.sinks.outbox import Outbox, OutboxItem, OutboxSender, SendResult

logger = logging.getLogger(__name__)

ALERTS = "alerts"


def cam_id_from(camera_id: str) -> int:
    """'cam_25' -> 25; если цифр нет — 0."""
    match = re.search(r"\d+", camera_id)
    return int(match.group()) if match else 0


class VisDashSink:
    def __init__(self, config: VisDashConfig, session: Any | None = None, idle_poll_sec: float = 5.0) -> None:
        self._cfg = config
        self._session = session or requests.Session()
        self._outbox = Outbox(config.outbox_dir, [ALERTS])
        if not config.ssl_verify:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self._sender = OutboxSender(
            self._outbox, self._send, name="VisDash",
            retry_max_delay_sec=config.retry_max_delay_sec, idle_poll_sec=idle_poll_sec,
        )

    def incident_type(self, violation_type: str, rule: str) -> int:
        types = self._cfg.incident_types
        for key in (violation_type, rule, "default"):
            if key in types:
                return int(types[key])
        return 0

    def publish(self, package: IncidentPackage) -> None:
        if package.status != "opened":
            return
        e = package.event
        image = next((f.data for f in package.files if f.kind == "frame_start"), None)
        if image is None:
            return
        payload = {
            "camId": self._cfg.cam_id if self._cfg.cam_id is not None else cam_id_from(e.camera_id),
            "roomId": self._cfg.room_id,
            "type": self.incident_type(e.violation_type, e.rule),
            "priority": self._cfg.priority,
            "status": 0,
            "date": package.started_at.astimezone(UTC).isoformat(),
        }
        meta = {"episode_id": e.episode_id, "violation_type": e.violation_type, "payload": payload}
        self._outbox.put(ALERTS, e.episode_id, meta, {"image.jpg": image})
        self._sender.wake()

    def _send(self, item: OutboxItem) -> SendResult:
        meta = item.meta
        # Старые элементы очереди содержат CamId/RoomId/Type/Priority.
        payload = {k[0].lower() + k[1:]: v for k, v in meta["payload"].items()}
        payload.setdefault("status", 0)
        try:
            payload["guid"] = str(UUID(meta["episode_id"]))
        except ValueError:
            pass  # Для старых не-UUID идентификатор назначает сервер.
        headers = {"Idempotency-Key": meta["episode_id"]}
        if key := os.getenv("PPE_VISDASH_API_KEY"):
            headers["X-API-Key"] = key
        elif token := os.getenv("PPE_VISDASH_BEARER_TOKEN"):
            headers["Authorization"] = f"Bearer {token}"
        try:
            response = self._session.post(
                self._cfg.url,
                files={"image": ("image.jpg", item.files()["image.jpg"].read_bytes(), "image/jpeg")},
                data={"incident": json.dumps(payload)},
                headers=headers,
                allow_redirects=False,
                timeout=self._cfg.timeout_sec,
                verify=self._cfg.ssl_verify,
            )
        except requests.RequestException:
            return SendResult.RETRY
        status = response.status_code
        if 200 <= status < 300:
            logger.info("VisDash: оповещение %s отправлено", meta["violation_type"])
            return SendResult.SENT
        if status == 429 or status >= 500:
            return SendResult.RETRY
        logger.error("VisDash отклонил оповещение (HTTP %d): %s", status, response.text[:200])
        return SendResult.REJECTED

    def close(self) -> None:
        self._sender.close(drain_sec=self._cfg.shutdown_drain_sec)
