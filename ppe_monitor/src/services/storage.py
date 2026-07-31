# src/services/storage.py
"""Сервис для асинхронного сохранения данных (Логи и Изображения)."""

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Кастомный JSON-энкодер для обработки numpy типов."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class StorageService:
    """Управляет сохранением кадров нарушений и JSON-логов."""

    def __init__(self, output_dir: str, log_file: str) -> None:
        """
        Инициализация путей и пула потоков.

        Args:
            output_dir: Директория для сохранения картинок.
            log_file: Путь к файлу JSON-логов.
        """
        self.output_dir = output_dir
        self.log_file = log_file     
        os.makedirs(self.output_dir, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="Storage")
        self.lock = threading.Lock()

    def save_violation(
            self, frame: np.ndarray, violation_data: Dict[str, Any]) -> None:
        """
        Отправляет задачу сохранения в отдельный поток.

        Args:
            frame: Кадры с отрисованными рамками (numpy array).
            violation_data: Словарь с данными о нарушении.
        """
        frame_copy = frame.copy()

        self.executor.submit(self._write_to_disk, frame_copy, violation_data)

    def _write_to_disk(self, frame: np.ndarray, data: Dict[str, Any]) -> None:
        """
        Физическая запись файлов на диск (выполняется в фоновом потоке).
        """
        try:
            filepath = data.get("frame_path")
            if not filepath:
                logger.error("No file path provided in violation data.")
                return

            cv2.imwrite(filepath, frame)

            with self.lock:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(data, cls=NumpyEncoder, ensure_ascii=False) + '\n')

            logger.info("[SAVED] Violation: %s -> %s", data.get('violation_type'), filepath)

        except Exception as e:
            logger.error("Error saving violation: %s", e)

    def shutdown(self) -> None:
        """Корректное завершение работы пула потоков."""
        logger.info("Shutting down Storage Service...")
        self.executor.shutdown(wait=True)
        logger.info("Storage Service stopped.")
