"""Отрисовка результатов анализа на кадре. Только рисует — ничего не решает."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import cv2
import numpy as np

from ppe_monitor.domain.analyzer import FrameAnalysis
from ppe_monitor.domain.models import Rule, Status, Zone

# BGR
COLOR_OK = (0, 200, 0)
COLOR_PENDING = (0, 165, 255)  # есть наблюдение, ещё не подтверждено
COLOR_CONFIRMED = (0, 0, 255)
COLOR_UNKNOWN = (160, 160, 160)
COLOR_PPE = (255, 120, 0)
COLOR_ZONE = (0, 0, 220)
COLOR_KEYPOINT = (255, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _pt(x: float, y: float) -> tuple[int, int]:
    return (int(round(x)), int(round(y)))


def _label(img: np.ndarray, text: str, x: float, y: float, color, scale: float = 0.5) -> None:
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(x), max(int(y), th + base)
    cv2.rectangle(img, (x, y - th - base), (x + tw, y + base // 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y - base // 2), FONT, scale, color, 1, cv2.LINE_AA)


def draw_zones(img: np.ndarray, zones: Sequence[Zone], alpha: float = 0.25) -> None:
    if not zones:
        return
    h, w = img.shape[:2]
    overlay = img.copy()
    polys = []
    for zone in zones:
        poly = np.array([_pt(x * w, y * h) for x, y in zone.polygon], dtype=np.int32)
        polys.append((zone, poly))
        cv2.fillPoly(overlay, [poly], COLOR_ZONE)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    for zone, poly in polys:
        cv2.polylines(img, [poly], True, COLOR_ZONE, 2, cv2.LINE_AA)
        _label(img, zone.name, *poly[0], COLOR_ZONE)


def draw(
    image: np.ndarray,
    analysis: FrameAnalysis,
    *,
    confirmed_track_ids: Collection[int] = (),
    zones: Sequence[Zone] = (),
    fps: float | None = None,
    min_keypoint_visibility: float = 0.3,
) -> np.ndarray:
    img = image.copy()
    draw_zones(img, zones)

    violating = analysis.violating()
    pending_ids = {o.track_id for o in violating}
    zone_names: dict[int, list[str]] = {}
    for o in violating:
        if o.rule == Rule.ZONE:
            zone_names.setdefault(o.track_id, []).append(o.subject)

    for person in analysis.persons:
        status = analysis.statuses.get(person.track_id)
        if person.track_id in confirmed_track_ids:
            color = COLOR_CONFIRMED
        elif person.track_id in pending_ids:
            color = COLOR_PENDING
        elif status is not None and status.status is Status.UNKNOWN:
            color = COLOR_UNKNOWN
        else:
            color = COLOR_OK

        x1, y1, x2, y2 = person.box
        cv2.rectangle(img, _pt(x1, y1), _pt(x2, y2), color, 2)

        text = f"ID {person.track_id}"
        if status is not None and status.missing:
            text += " no " + "+".join(sorted(status.missing))
        if status is not None and status.undetermined:
            text += " ?" + "+".join(sorted(status.undetermined))
        if person.track_id in zone_names:
            text += " in " + ",".join(zone_names[person.track_id])
        _label(img, text, x1, y1 - 4, color)

        if person.keypoints is not None:
            for x, y, vis in person.keypoints:
                if vis >= min_keypoint_visibility:
                    cv2.circle(img, _pt(x, y), 2, COLOR_KEYPOINT, -1)

        for item in person.ppe:
            px1, py1, px2, py2 = item.box
            cv2.rectangle(img, _pt(px1, py1), _pt(px2, py2), COLOR_PPE, 1)
            _label(img, item.class_name, px1, py2 + 14, COLOR_PPE, 0.4)

    if fps is not None:
        _label(img, f"FPS {fps:.1f}", 10, 30, COLOR_OK, 0.8)
    return img


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Не удалось закодировать JPEG")
    return buf.tobytes()
