"""Подстановка 'auto' под железо: CPU, GTX 1050 (4 ГБ), RTX 4060 (8 ГБ), Apple, явные значения."""

import logging

import pytest

from ppe_monitor import hardware
from ppe_monitor.config import AppConfig, HardwareConfig, ModelsConfig, SourceConfig
from ppe_monitor.hardware import HardwareInfo

CPU = HardwareInfo(8, 32.0, "cpu", "-", 0.0)
GTX_1050 = HardwareInfo(4, 16.0, "cuda", "GeForce GTX 1050", 4.0)
RTX_4060 = HardwareInfo(8, 32.0, "cuda", "GeForce RTX 4060", 8.0)
APPLE = HardwareInfo(8, 16.0, "mps", "Apple Silicon", 0.0)


def resolve(hw, **sections):
    return hardware.resolve(AppConfig(**sections), hw)


@pytest.mark.parametrize(
    "hw,device,half,imgsz,every_n",
    [
        (CPU, "cpu", False, 640, 3),
        (GTX_1050, "cuda", True, 960, 2),
        (RTX_4060, "cuda", True, 1280, 1),
        (APPLE, "mps", False, 640, 2),  # раньше на Mac выбиралась cuda и всё падало
    ],
)
def test_auto_profiles(hw, device, half, imgsz, every_n):
    rt = resolve(hw)
    assert (rt.device, rt.half, rt.imgsz, rt.process_every_n_frames) == (device, half, imgsz, every_n)
    assert rt.torch_threads is None


def test_explicit_values_win():
    rt = resolve(
        RTX_4060,
        hardware=HardwareConfig(device="cpu", half=False, torch_threads=4),
        models=ModelsConfig(imgsz=768),
        source=SourceConfig(process_every_n_frames=5),
    )
    assert (rt.device, rt.half, rt.torch_threads, rt.imgsz, rt.process_every_n_frames) == ("cpu", False, 4, 768, 5)


def test_half_on_cpu_is_disabled_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        rt = resolve(CPU, hardware=HardwareConfig(half=True))
    assert rt.half is False
    assert "half" in caplog.text


def test_detect_on_this_machine():
    hw = hardware.detect()
    assert hw.cpu_cores >= 1
    assert hw.accelerator in {"cpu", "cuda", "mps"}
    assert (hw.vram_gb > 0) == (hw.accelerator == "cuda")
