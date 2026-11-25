"""Точка входа и «сборка» приложения (composition root).

Это единственное место, где известно, какие конкретные классы используются:
здесь читается конфиг, создаются объекты и соединяются друг с другом.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

from ppe_monitor import __version__
from ppe_monitor.config import AppConfig, ConfigError, LoggingConfig, load_config, load_zones

logger = logging.getLogger("ppe_monitor")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ppe-monitor", description="Мониторинг СИЗ на видео")
    p.add_argument("--config", type=Path, default=None, help="YAML-конфиг (по умолчанию config.yaml проекта)")
    p.add_argument("--source", help="индекс камеры, путь к видео или rtsp/http-адрес")
    p.add_argument("--camera-id", help="идентификатор камеры в журналах и API")
    p.add_argument("--zones", type=Path, help="файл запретных зон (YAML)")
    p.add_argument("--resize", type=int, help="ширина кадра перед обработкой")
    p.add_argument("--device", help="cpu | cuda | cuda:0 | mps | auto")
    p.add_argument("--no-display", action="store_true", help="без окна предпросмотра")
    p.add_argument("--no-api", action="store_true", help="не отправлять по сети (приёмник и VisDash)")
    p.add_argument("--max-frames", type=int, help="остановиться после N обработанных кадров")
    p.add_argument("-v", "--verbose", action="store_true", help="подробные логи")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args(argv)


def overrides_from(args: argparse.Namespace) -> dict:
    o = {}
    if args.source is not None:
        o["source.uri"] = args.source
    if args.camera_id:
        o["camera_id"] = args.camera_id
    if args.zones:
        o["rules.zones_file"] = str(args.zones.resolve())
    if args.resize:
        o["source.resize_width"] = args.resize
    if args.device:
        o["hardware.device"] = args.device
    if args.no_display:
        o["display.enabled"] = False
    if args.no_api:
        o["receiver.enabled"] = False
        o["visdash.enabled"] = False
    return o


def setup_logging(cfg: LoggingConfig, verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if cfg.file:
        cfg.file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(cfg.file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else cfg.level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    for noisy in ("ultralytics", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run(config: AppConfig, max_frames: int | None = None) -> int:
    # Тяжёлые импорты — здесь, а не в начале модуля: `--help` и ошибки конфига работают мгновенно.
    from ppe_monitor import hardware
    from ppe_monitor.domain.analyzer import Analyzer
    from ppe_monitor.domain.episode_tracker import EpisodeTracker
    from ppe_monitor.domain.zones import foot_point
    from ppe_monitor.pipeline.detector import YoloDetector
    from ppe_monitor.pipeline.evidence import EvidenceCollector
    from ppe_monitor.pipeline.projection import FloorProjector
    from ppe_monitor.pipeline.runner import Pipeline
    from ppe_monitor.pipeline.source import open_source
    from ppe_monitor.sinks.base import Broadcast
    from ppe_monitor.sinks.files import FileSink
    from ppe_monitor.sinks.receiver import ReceiverSink
    from ppe_monitor.sinks.visdash import VisDashSink
    from ppe_monitor.ui.display import Window

    logger.info("PPE Monitor %s, камера %s, источник %s", __version__, config.camera_id, config.source.uri)

    hw = hardware.detect()
    rt = hardware.resolve(config, hw)
    hardware.log_summary(hw, rt)
    if rt.torch_threads:
        import torch

        torch.set_num_threads(rt.torch_threads)

    zones = load_zones(config.rules.zones_file) if config.rules.zones_file else []
    if zones:
        logger.info("Запретные зоны: %s", ", ".join(z.name for z in zones))

    m = config.models
    detector = YoloDetector(
        m.pose, m.ppe, device=rt.device, imgsz=rt.imgsz, half=rt.half,
        person_conf=m.person_conf, ppe_conf=m.ppe_conf, iou=m.iou,
        tracker=m.tracker, ignore_classes=m.ignore_classes,
    )
    policy = config.rules.to_policy()
    unknown = {c.lower() for c in policy.required} - {c.lower() for c in detector.ppe_classes}
    if unknown:
        raise ConfigError(
            f"Модель {m.ppe.name} не умеет находить {sorted(unknown)}. "
            f"Её классы: {sorted(detector.ppe_classes)}. Исправьте rules.required_ppe или models.ppe."
        )

    sinks = []
    if config.storage.enabled:
        sinks.append(FileSink(config.storage.incidents_dir, config.storage.journal_file))
    if config.receiver.enabled:
        sinks.append(ReceiverSink(config.receiver))
    if config.visdash.enabled:
        sinks.append(VisDashSink(config.visdash))
    broadcast = Broadcast(sinks)

    locate = None
    cal = config.calibration
    if cal.enabled:
        projector = FloorProjector(cal.image_points, cal.world_points)

        def locate(person, frame_size):
            foot = foot_point(person, frame_size, policy.min_keypoint_visibility, policy.edge_margin_px)
            return None if foot is None else projector.to_floor(*foot)

    window = Window(config.display.window_name) if config.display.enabled else None
    source = open_source(config.source, rt.process_every_n_frames)

    stop = threading.Event()

    def on_signal(signum, _frame):
        logger.info("Получен сигнал %s — завершаю", signum)
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    evidence = EvidenceCollector(
        config.evidence,
        broadcast.publish,
        source_name=source.name,
        model_info={"pose": m.pose.name, "ppe": m.ppe.name, "imgsz": rt.imgsz},
        min_keypoint_visibility=policy.min_keypoint_visibility,
        locate=locate,
    )
    pipeline = Pipeline(
        detector,
        Analyzer(policy, zones),
        EpisodeTracker(
            config.camera_id, config.timings(), reid_max_distance=config.tracking.reid_max_distance
        ),
        evidence,
        show=window.show if window else None,
    )
    try:
        stats = pipeline.run(source, stop, max_frames)
    finally:
        source.close()
        if window:
            window.close()
        evidence.close()  # дождаться клипов и последних пакетов
        broadcast.close()
    logger.info(
        "Готово: %d кадров за %.1f с (%.1f к/с), нарушений открыто: %d, закрыто: %d",
        stats.frames, stats.seconds, stats.fps, stats.opened, stats.closed,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    # UTF-8 одинаково в консоли Windows и при перенаправлении вывода в файл/пайп.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    try:
        config = load_config(args.config, overrides_from(args))
    except ConfigError as e:
        print(f"Ошибка конфигурации: {e}", file=sys.stderr)
        return 2

    setup_logging(config.logging, args.verbose)
    try:
        return run(config, args.max_frames)
    except (ConfigError, FileNotFoundError, RuntimeError) as e:
        logger.error("%s", e)
        return 1
