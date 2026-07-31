# src/services/api_client.py
"""Сервис для отправки нарушений на сервер через REST API."""

import json
import logging
import os
import re
import threading
from queue import Queue, Full
from typing import Any, Dict

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Энкодер для сериализации numpy-типов в JSON."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def extract_cam_id(camera_id_str: str) -> int:
    """
    Извлекает числовой ID из строки вида 'cam_25' → 25.

    Args:
        camera_id_str: Строка с идентификатором камеры.

    Returns:
        int: Числовой ID или 0, если не удалось извлечь.
    """
    match = re.search(r'\d+', camera_id_str)
    return int(match.group()) if match else 0


class APIClientService:
    """
    Асинхронный клиент для отправки нарушений на сервер.

    Использует очередь задач и фоновый поток, чтобы не блокировать
    основной цикл обработки видео.
    """

    def __init__(
        self,
        api_url: str,
        room_id: int = 12,
        priority: int = 1,
        ssl_verify: bool = False,
        max_queue_size: int = 50,
        timeout: float = 10.0
    ) -> None:
        """
        Инициализация клиента.

        Args:
            api_url: Полный URL эндпоинта (например, 'https://.../send-incident').
            room_id: ID помещения для всех инцидентов.
            incident_type: Тип инцидента (2 = нарушение СИЗ).
            priority: Приоритет (1 = высокий).
            ssl_verify: Проверка SSL-сертификата (False для self-signed).
            max_queue_size: Максимальный размер очереди задач.
            timeout: Таймаут HTTP-запросов в секундах.
        """
        self.api_url = api_url
        self.room_id = room_id
        self.priority = priority
        self.ssl_verify = ssl_verify
        self.timeout = timeout

        self.task_queue: Queue[Dict[str, Any]] = Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()

        # Настройка сессии с ретраями
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self._worker = threading.Thread(
            target=self._process_queue,
            daemon=True,
            name="API-Worker"
        )
        self._worker.start()

        logger.info("🌐 APIClientService initialized: %s", api_url)

    def _incident_type_to_int(self, violation_data: Dict[str, Any]) -> int:
        """
        Приводим тип нарушения к int
        """
        # violation = violation_data.get("violation_type")

        # if violation == "helmet":
        #     return 1
        # elif violation == "vest":
        #     return 2
        # elif violation == "helmet_vest":
        #     return 3
        return 0

    def send_violation(
            self, frame: np.ndarray,
            violation_data: Dict[str, Any],
            output_dir: str) -> bool:
        """
        Добавляет задачу отправки нарушения в очередь.

        Args:
            frame: Кадр с отрисованными рамками (для сохранения).
            violation_ Данные нарушения (frame_path, camera_id, etc.).
            output_dir: Директория для временного сохранения изображения.

        Returns:
            bool: True если задача добавлена, False если очередь переполнена.
        """
        try:
            task = {
                "frame": frame.copy(),
                "violation_data": violation_data.copy(),
                "output_dir": output_dir
            }
            self.incident_type = self._incident_type_to_int(violation_data)
            self.task_queue.put_nowait(task)
            return True
        except Full:
            logger.warning("API queue is full. Dropping violation for track_id=%s",
                        violation_data.get("track_id"))
            return False

    def _process_queue(self) -> None:
        """Фоновый поток: обрабатывает очередь задач."""
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=1.0)
                self._send_to_server(task)
                self.task_queue.task_done()
            except Exception:
                pass

    def _send_to_server(self, task: Dict[str, Any]) -> None:
        """
        Выполняет сохранение файла и отправку на сервер.

        Args:
            task: Словарь с frame, violation_data, output_dir.
        """
        frame = task["frame"]
        data = task["violation_data"]
        output_dir = task["output_dir"]

        try:
            os.makedirs(output_dir, exist_ok=True)
            filepath = data.get("frame_path")
            if not filepath:
                logger.error("No frame_path in violation data")
                return

            cv2.imwrite(filepath, frame)

            camera_id_str = data.get("camera_id", "cam_01")
            camera_id_num = extract_cam_id(camera_id_str)

            incident_json = json.dumps({
                "CamId": camera_id_num,
                "RoomId": self.room_id,
                "Type": self.incident_type,
                "Priority": self.priority
            }, cls=NumpyEncoder)

            with open(filepath, 'rb') as img_file:
                files = {
                    'image': ('image.jpg', img_file, 'image/jpeg')
                }
                form_data = {
                    'incident': incident_json
                }

                response = self.session.post(
                    self.api_url,
                    files=files,
                    data=form_data,
                    timeout=self.timeout,
                    verify=self.ssl_verify
                )

                if response.status_code in (200, 201):
                    logger.info("[API] ✅ Sent: %s -> %s", 
                               data.get('violation_type'), filepath)
                else:
                    logger.warning("[API] ⚠️ Server returned %d: %s", 
                                  response.status_code, response.text[:200])

        except requests.RequestException as e:
            logger.error("[API] ❌ Request failed: %s", e)
        except Exception as e:
            logger.error("[API] ❌ Unexpected error: %s", e)

    def shutdown(self, timeout: float = 10.0) -> None:
        """
        Корректно завершает работу: дожидается отправки очереди.

        Args:
            timeout: Максимальное время ожидания завершения в секундах.
        """
        logger.info("🛑 Shutting down APIClientService...")
        self.stop_event.set()

        if not self.task_queue.join():
            logger.info("✅ All pending API tasks completed")

        self._worker.join(timeout=timeout)
        self.session.close()
        logger.info("✅ APIClientService stopped.")
