# src/services/detector.py
"""Сервис для работы с YOLO моделями (Детекция и Трекинг)."""

import logging
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

from src.core.models import Person, PPEItem

logger = logging.getLogger(__name__)


class DetectorService:
    """Сервис управления моделями YOLO для детекции людей и СИЗ."""

    def __init__(self, pose_model_path: str, ppe_model_path: str,
            device: str = "cpu", imgsz: int = 640, half: bool = False) -> None:
        self.device = device
        self.half = half and device != "cpu"  # half=True только на GPU
        self.imgsz = imgsz
        
        logger.info("Loading models: device=%s, half=%s", self.device, self.half)
        self.pose_model = YOLO(pose_model_path)
        self.ppe_model = YOLO(ppe_model_path)
        logger.info("Models loaded successfully.")

    def process_frame(
        self,
        frame: np.ndarray,
        conf: float = 0.5,
        iou: float = 0.5
    ) -> Tuple[List[Person], List[PPEItem]]:
        pose_results = self.pose_model.track(
            frame, conf=conf, iou=iou, persist=True, verbose=False,
            device=self.device, half=self.half,
            imgsz=self.imgsz
        )[0]

        ppe_results = self.ppe_model(
            frame, conf=conf, verbose=False,
            device=self.device, half=self.half,
            imgsz=self.imgsz
        )[0]

        persons = []
        ppe_items = []

        if pose_results.boxes is not None and len(pose_results.boxes) > 0:
            boxes = pose_results.boxes
            ids = boxes.id.cpu().numpy() if boxes.id is not None else None
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            keypoints_data = pose_results.keypoints.data.cpu().numpy()

            for i in range(len(boxes)):
                if ids is None or ids[i] is None:
                    continue

                track_id = int(ids[i])
                x1, y1, x2, y2 = [int(x) for x in xyxy[i]]

                persons.append(Person(
                    track_id=track_id,
                    box=[x1, y1, x2, y2],
                    confidence=float(confs[i]),
                    keypoints=keypoints_data[i]
                ))

        if ppe_results.boxes is not None and len(ppe_results.boxes) > 0:
            boxes = ppe_results.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                class_id = int(classes[i])
                class_name = self.ppe_model.names[class_id]

                if class_name.lower() == 'person':
                    continue

                x1, y1, x2, y2 = [int(x) for x in xyxy[i]]
                ppe_items.append(PPEItem(
                    class_name=class_name,
                    box=[x1, y1, x2, y2],
                    confidence=float(confs[i])
                ))

        return persons, ppe_items
