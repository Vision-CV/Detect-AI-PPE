#!/usr/bin/env python3
"""Редактор запретных зон: рисуем многоугольники мышкой на кадре из видео.

    python tools/draw_zones.py --source test/test2.mp4 --out zones.yaml [--frame 100]

Управление:
    ЛКМ        — добавить вершину
    ПКМ / z    — удалить последнюю вершину
    Enter      — замкнуть многоугольник (имя зоны спросят в консоли)
    d          — удалить последнюю зону
    s          — сохранить и выйти
    q / Esc    — выйти без сохранения

Координаты сохраняются в долях кадра (0..1), поэтому зоны не зависят от разрешения.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppe_monitor.config import load_zones, save_zones  # noqa: E402
from ppe_monitor.domain.models import Zone  # noqa: E402
from ppe_monitor.ui.renderer import COLOR_ZONE, draw_zones  # noqa: E402

WINDOW = "Zones editor"


def read_frame(source: str, index: int):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Не удалось прочитать кадр {index} из {source}")
    return frame


def main() -> None:
    p = argparse.ArgumentParser(description="Редактор запретных зон")
    p.add_argument("--source", required=True, help="видео, картинка или индекс камеры")
    p.add_argument("--out", type=Path, default=Path("zones.yaml"))
    p.add_argument("--frame", type=int, default=0, help="номер кадра для фона")
    p.add_argument("--resize", type=int, help="та же ширина, что и при запуске мониторинга")
    args = p.parse_args()

    image = cv2.imread(args.source) if Path(args.source).suffix.lower() in {".jpg", ".png"} else None
    if image is None:
        image = read_frame(args.source, args.frame)
    if args.resize:
        h, w = image.shape[:2]
        image = cv2.resize(image, (args.resize, round(h * args.resize / w)))
    h, w = image.shape[:2]

    zones: list[Zone] = load_zones(args.out) if args.out.is_file() else []
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(__doc__)

    while True:
        canvas = image.copy()
        draw_zones(canvas, zones)
        for i, pt in enumerate(points):
            cv2.circle(canvas, pt, 4, COLOR_ZONE, -1)
            if i:
                cv2.line(canvas, points[i - 1], pt, COLOR_ZONE, 2)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 10):  # Enter
            if len(points) < 3:
                print("Нужно минимум 3 точки")
                continue
            name = input("Имя зоны (латиницей, например crane): ").strip() or f"zone{len(zones) + 1}"
            zones.append(Zone(name, tuple((x / w, y / h) for x, y in points)))
            points.clear()
            print(f"Добавлена зона '{name}'. Всего: {len(zones)}")
        elif key == ord("z") and points:
            points.pop()
        elif key == ord("d") and zones:
            print(f"Удалена зона '{zones.pop().name}'")
        elif key == ord("s"):
            save_zones(args.out, zones)
            print(f"Сохранено {len(zones)} зон в {args.out}")
            break
        elif key in (ord("q"), 27):
            print("Выход без сохранения")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
