# src/utils/hardware.py
"""Модуль автоматического определения железа и генерации оптимальных настроек."""

import logging
import os
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def _detect_cpu_and_ram() -> Dict[str, Any]:
    """Определяет физические ядра CPU и объём RAM."""
    if HAS_PSUTIL:
        phys_cores = psutil.cpu_count(logical=False) or os.cpu_count()
        ram_gb = psutil.virtual_memory().total / (1024**3)
    else:
        phys_cores = os.cpu_count()
        ram_gb = 8.0  # Консервативный fallback
        logger.warning("psutil не установлен. Используются приближённые значения RAM.")

    return {"cpu_cores": phys_cores, "ram_gb": round(ram_gb, 1)}


def _detect_gpu() -> Dict[str, Any]:
    """Определяет доступность и характеристики GPU."""
    if torch.cuda.is_available():
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        return {
            "type": "nvidia",
            "name": torch.cuda.get_device_name(0),
            "vram_gb": round(vram_bytes / (1024**3), 1),
            "available": True
        }

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {"type": "apple_silicon",
                "name": "Apple M-Series",
                "vram_gb": 8.0, "available": True}
    
    return {"type": "cpu",
            "name": "CPU / Integrated",
            "vram_gb": 0.0, "available": False}


def get_hardware_profile() -> Dict[str, Any]:
    """
    Возвращает оптимальные настройки под текущее железо.
    Профили: high, medium, low.
    """
    cpu_ram = _detect_cpu_and_ram()
    gpu = _detect_gpu()

    cores = cpu_ram["cpu_cores"]
    ram = cpu_ram["ram_gb"]
    vram = gpu["vram_gb"]

    if vram >= 6.0 and cores >= 6 and ram >= 16:
        profile = "high"
    elif (vram >= 4.0) or (cores >= 4 and ram >= 8):
        profile = "medium"
    else:
        profile = "low"

    configs = {
        "high": {
            "device": "cuda",
            "half": True,
            "imgsz": 1280,
            "torch_threads": 0,
            "skip_frames": 3,
            "workers": 4,
            "profile_desc": "High-End (Dedicated GPU + 16GB+ RAM)",
            "PPE_MODEL": "models_weights/PPE_M_YOLO_1280.onnx"
        },
        "medium": {
            "device": "cuda" if gpu["available"] else "cpu",
            "half": gpu["available"],
            "imgsz": 960,
            "torch_threads": max(4, cores // 2),
            "skip_frames": 4,
            "workers": 2,
            "profile_desc": "Balanced (Mid GPU or Strong CPU)",
            "PPE_MODEL": "models_weights/PPE_M_YOLO_960.onnx"
        },
        "low": {
            "device": "cpu",
            "half": False,
            "imgsz": 640,
            "torch_threads": 2,
            "skip_frames": 5,
            "workers": 0,
            "profile_desc": "Low-End / CPU / Integrated GPU",
            "PPE_MODEL": "models_weights/PPE_M_YOLO_640.onnx"
        }
    }

    profile_config = configs[profile]
    profile_config["hardware"] = {**cpu_ram, **gpu}

    logger.info("="*50)
    logger.info("🔍 HARDWARE AUTO-OPTIMIZER")
    logger.info(f"   Profile: {profile_config['profile_desc']}")
    logger.info(f"   CPU: {cores} cores | RAM: {ram:.1f} GB")
    logger.info(f"   GPU: {gpu['name']} | VRAM: {gpu['vram_gb']:.1f} GB")
    logger.info(f"   → device={profile_config['device']}, half={profile_config['half']}")
    logger.info(f"   → imgsz={profile_config['imgsz']}, skip={profile_config['skip_frames']}")
    logger.info(f"   → torch_threads={profile_config['torch_threads']}, workers={profile_config['workers']}")
    logger.info("="*50)

    return profile_config


__all__ = ["get_hardware_profile"]
