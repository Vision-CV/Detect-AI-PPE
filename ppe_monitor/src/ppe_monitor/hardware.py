"""Определение железа и выбор значений для параметров со значением 'auto'.

Железо только ПОДСКАЗЫВАЕТ значения по умолчанию. Всё, что задано в конфиге
явно, используется как есть. Выбор файла модели — всегда решение пользователя.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from ppe_monitor.config import AUTO, AppConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareInfo:
    cpu_cores: int
    ram_gb: float
    accelerator: str  # "cuda" | "mps" | "cpu"
    gpu_name: str
    vram_gb: float


@dataclass(frozen=True)
class Runtime:
    """Итоговые параметры запуска после подстановки 'auto'."""

    device: str
    half: bool
    torch_threads: int | None  # None — оставить по умолчанию torch
    imgsz: int
    process_every_n_frames: int


def detect() -> HardwareInfo:
    import torch  # импорт здесь: модуль конфигурации не должен тянуть torch

    try:
        import psutil

        cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        ram_gb = psutil.virtual_memory().total / 1024**3
    except ImportError:
        cores, ram_gb = os.cpu_count() or 1, 0.0

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return HardwareInfo(cores, ram_gb, "cuda", props.name, props.total_memory / 1024**3)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return HardwareInfo(cores, ram_gb, "mps", "Apple Silicon", 0.0)
    return HardwareInfo(cores, ram_gb, "cpu", "-", 0.0)


def resolve(config: AppConfig, hw: HardwareInfo) -> Runtime:
    hc = config.hardware
    device = hw.accelerator if hc.device == AUTO else hc.device
    on_cuda = device.startswith("cuda")
    strong_gpu = on_cuda and hw.vram_gb >= 6

    half = on_cuda if hc.half == AUTO else bool(hc.half)
    if half and not on_cuda:
        logger.warning("half=True имеет смысл только на CUDA — отключаю")
        half = False

    if hc.torch_threads == AUTO:
        threads = None
    else:
        threads = int(hc.torch_threads)

    imgsz = config.models.imgsz
    if imgsz == AUTO:
        imgsz = 1280 if strong_gpu else 960 if on_cuda else 640

    every_n = config.source.process_every_n_frames
    if every_n == AUTO:
        every_n = 1 if strong_gpu else 2 if device != "cpu" else 3

    return Runtime(device, half, threads, int(imgsz), int(every_n))


def log_summary(hw: HardwareInfo, rt: Runtime) -> None:
    logger.info(
        "Железо: CPU %d ядер, RAM %.1f ГБ, ускоритель %s (%s, %.1f ГБ)",
        hw.cpu_cores, hw.ram_gb, hw.accelerator, hw.gpu_name, hw.vram_gb,
    )
    logger.info(
        "Параметры: device=%s half=%s imgsz=%d каждый %d-й кадр, torch_threads=%s",
        rt.device, rt.half, rt.imgsz, rt.process_every_n_frames, rt.torch_threads or "по умолчанию",
    )
