"""Детекция людей (с трекингом и позой) и СИЗ.

`Detector` — контракт: любой класс с методом `detect` подходит конвейеру.
Это позволяет подставить фейковый детектор в тестах или попробовать
другую модель (например, одну вместо двух), не трогая остальной код.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from ppe_monitor.domain.models import Person, PPEItem

logger = logging.getLogger(__name__)


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> tuple[list[Person], list[PPEItem]]: ...


class YoloDetector:
    def __init__(
        self,
        pose_model: Path,
        ppe_model: Path,
        *,
        device: str,
        imgsz: int,
        half: bool,
        person_conf: float,
        ppe_conf: float,
        iou: float,
        tracker: str,
        ignore_classes: tuple[str, ...],
    ) -> None:
        from ultralytics import YOLO

        for path in (pose_model, ppe_model):
            if not path.is_file():
                raise FileNotFoundError(f"Файл модели не найден: {path}")
        if ppe_model.suffix in {".onnx", ".engine"}:
            logger.info("Модель %s экспортирована с фиксированным размером входа — imgsz должен ему соответствовать", ppe_model.name)

        logger.info("Загрузка моделей: %s, %s (device=%s)", pose_model.name, ppe_model.name, device)
        self._pose = YOLO(str(pose_model))
        self._ppe = YOLO(str(ppe_model))
        self._common = {"device": device, "imgsz": imgsz, "verbose": False}
        # half — параметр для PyTorch-весов; у ONNX/TensorRT точность зашита при экспорте.
        # Передаём его только когда он нужен: в новых ultralytics сам ключ помечен устаревшим.
        self._pose_extra = {"half": True} if half and pose_model.suffix == ".pt" else {}
        self._ppe_extra = {"half": True} if half and ppe_model.suffix == ".pt" else {}
        self._person_conf = person_conf
        self._ppe_conf = ppe_conf
        self._iou = iou
        self._tracker = tracker
        self._ignore = {c.lower() for c in ignore_classes}

    @property
    def ppe_classes(self) -> set[str]:
        return set(self._ppe.names.values())

    def detect(self, image: np.ndarray) -> tuple[list[Person], list[PPEItem]]:
        pose = self._pose.track(
            image, persist=True, tracker=self._tracker, conf=self._person_conf,
            iou=self._iou, **self._pose_extra, **self._common,
        )[0]
        ppe = self._ppe(image, conf=self._ppe_conf, iou=self._iou, **self._ppe_extra, **self._common)[0]
        return self._persons(pose), self._ppe_items(ppe)

    @staticmethod
    def _persons(result) -> list[Person]:
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []  # трекер ещё не присвоил id ни одному человеку
        ids = boxes.id.int().tolist()
        xyxy = boxes.xyxy.tolist()
        confs = boxes.conf.tolist()
        kps = result.keypoints.data.tolist() if result.keypoints is not None else [None] * len(ids)
        return [
            Person(
                track_id=tid,
                box=tuple(box),
                confidence=conf,
                keypoints=tuple(tuple(p) for p in kp) if kp is not None else None,
            )
            for tid, box, conf, kp in zip(ids, xyxy, confs, kps)
        ]

    def _ppe_items(self, result) -> list[PPEItem]:
        boxes = result.boxes
        if boxes is None:
            return []
        items = []
        for cls_id, box, conf in zip(boxes.cls.int().tolist(), boxes.xyxy.tolist(), boxes.conf.tolist()):
            name = self._ppe.names[cls_id].lower()
            if name not in self._ignore:
                items.append(PPEItem(name, tuple(box), conf))
        return items
