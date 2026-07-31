#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPE Monitoring System — Main Entry Point.

This module orchestrates the detection, tracking, and violation logging pipeline.
"""

import argparse
import logging
import signal
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Set

import cv2
import torch

from config import settings
from src.core.association import associate_ppe
from src.core.compliance import check_compliance
from src.services.detector import DetectorService
from src.services.projector import ProjectorService
from src.services.api_client import APIClientService
from src.services.storage import StorageService
from src.ui.renderer import draw_frame


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Настраивает логирование для всего приложения."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("ppe_monitor.log", encoding="utf-8", mode="a")
        ]
    )


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(description="PPE Monitoring System")
    parser.add_argument("--source", type=str, default="0", help="Источник видео (0 или путь к файлу)")
    parser.add_argument("--verbose", action="store_true", help="Включить подробное логирование")
    parser.add_argument("--no-display", action="store_true", help="Отключить вывод окна OpenCV")
    parser.add_argument("--resize", type=int, default=None, help="Изменить ширину кадра (сохраняя пропорции)")
    return parser.parse_args()


def main() -> None:
    """Точка входа в приложение."""
    args = parse_args()
    setup_logging(args.verbose)

    if settings.TORCH_THREADS > 0:
        torch.set_num_threads(settings.TORCH_THREADS)
        os.environ["OMP_NUM_THREADS"] = str(settings.TORCH_THREADS)
        os.environ["MKL_NUM_THREADS"] = str(settings.TORCH_THREADS)
        logger.info(f"🔧 PyTorch threads limited to: {settings.TORCH_THREADS}")

    logger.info("🚀 Starting PPE Monitoring System v1.0")

    if settings.TORCH_THREADS > 0:
        torch.set_num_threads(settings.TORCH_THREADS)
    logger.info("PyTorch threads set to: %d", torch.get_num_threads())

    detector = DetectorService(
        settings.POSE_MODEL,
        settings.PPE_MODEL,
        device=settings.DEVICE,
        imgsz=settings.IMG_SIZE,
        half=settings.HALF_PRECISION
    )
    storage = StorageService(settings.OUTPUT_DIR, settings.LOG_FILE)
    api_client = APIClientService(
        api_url=settings.API_INCIDENT_URL,
        room_id=settings.API_ROOM_ID,
        priority=settings.API_PRIORITY,
        ssl_verify=settings.API_SSL_VERIFY
    )

    projector = None
    if settings.CALIB_IMAGE_PTS and settings.CALIB_WORLD_PTS:
        try:
            projector = ProjectorService(settings.CALIB_IMAGE_PTS, settings.CALIB_WORLD_PTS)
        except ValueError as e:
            logger.warning("⚠️ Projector init failed: %s", e)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("❌ Failed to open video source: %s", source)
        sys.exit(1)

    # track_id -> {first_seen, last_reported, lost_since}
    violation_states: Dict[int, Dict[str, float]] = {}
    confirmed_ids: Set[int] = set()

    stop_event = False

    def signal_handler(sig, frame):
        nonlocal stop_event
        logger.info("🛑 Shutdown signal received...")
        stop_event = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    frame_count = 0
    start_time = time.time()
    last_vis_frame = None

    try:
        while not stop_event:
            ret, frame = cap.read()
            if not ret:
                logger.info("🏁 End of video stream")
                break

            frame_count += 1

            if frame_count % settings.PROCESS_EVERY_N_FRAMES != 0:
                display_frame = last_vis_frame if last_vis_frame is not None else frame

                if not args.no_display:
                    cv2.imshow("PPE Monitoring", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop_event = True
                        break
                continue

            if args.resize:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (args.resize, int(h * args.resize / w)))

            persons, ppe_items = detector.process_frame(
                frame,
                conf=settings.CONF_THRESHOLD
            )

            persons = associate_ppe(persons, ppe_items)

            violations = check_compliance(persons)

            current_ts = time.time()
            current_violating_ids = {v.person.track_id for v in violations}

            for v in violations:
                tid = v.person.track_id
                if tid not in violation_states:
                    violation_states[tid] = {
                        "first_seen": current_ts,
                        "last_reported": 0.0,
                        "lost_since": None
                    }
                violation_states[tid]["lost_since"] = None

            for tid in list(violation_states.keys()):
                if tid not in current_violating_ids:
                    state = violation_states[tid]
                    if state["lost_since"] is None:
                        state["lost_since"] = current_ts
                    elif current_ts - state["lost_since"] > settings.TRACK_LOSS_GRACE_SEC:
                        del violation_states[tid]

            confirmed_ids.clear()
            for tid, state in violation_states.items():
                if state["lost_since"] is not None:
                    continue

                elapsed = current_ts - state["first_seen"]
                cooldown = current_ts - state["last_reported"]

                if elapsed >= settings.VIOLATION_CONFIRMATION_SEC and cooldown >= settings.VIOLATION_COOLDOWN_SEC:
                    confirmed_ids.add(tid)

                    v_data = next(v for v in violations if v.person.track_id == tid)
                    missing_str = "_".join(sorted(v_data.missing_items))
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"{ts_str}_{settings.CAMERA_ID}_ID{tid}_missing_{missing_str}.jpg"
                    filepath = str(Path(settings.OUTPUT_DIR) / filename)

                    pos_2d = None
                    if projector:
                        box = v_data.person.box
                        foot_x = (box[0] + box[2]) / 2
                        foot_y = box[3]
                        mx, mz = projector.pixel_to_meter(foot_x, foot_y)
                        if mx is not None:
                            pos_2d = {"x": round(mx, 2), "z": round(mz, 2)}

                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "camera_id": settings.CAMERA_ID,
                        "track_id": tid,
                        "violation_type": missing_str,
                        "person_box": v_data.person.box,
                        "confidence": v_data.person.confidence,
                        "frame_path": filepath,
                        "present_classes": [p.class_name for p in v_data.person.assigned_ppe],
                        "position_2d_meters": pos_2d,
                    }

                    ann_frame = draw_frame(frame, persons, [v_data], confirmed_ids={tid}, projector=projector)
                    storage.save_violation(ann_frame, log_entry)
                    api_client.send_violation(ann_frame, log_entry, settings.OUTPUT_DIR)

                    state["last_reported"] = current_ts
                    state["first_seen"] = current_ts

            if not args.no_display:
                elapsed_total = time.time() - start_time
                fps = frame_count / elapsed_total if elapsed_total > 0 else 0.0

                vis_frame = draw_frame(frame, persons, violations, confirmed_ids, projector, fps)

                last_vis_frame = vis_frame

                cv2.imshow("PPE Monitoring", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event = True
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event = True

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        logger.info("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        storage.shutdown()
        api_client.shutdown()
        logger.info("Shutdown complete. Processed %d frames.", frame_count)


if __name__ == "__main__":
    main()
