from pathlib import Path

import pytest

from ppe_monitor.config import DEFAULT_CONFIG, ConfigError, load_config, load_zones, save_zones
from ppe_monitor.domain.models import Rule, Zone


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_project_config_is_valid():
    config = load_config(DEFAULT_CONFIG)
    assert config.rules.required_ppe == ("helmet", "vest")
    assert set(config.timings()) == {Rule.PPE, Rule.ZONE}


def test_empty_config_gives_defaults(tmp_path):
    config = load_config(write(tmp_path, ""))
    assert config.models.ppe_conf == 0.35


def test_relative_paths_resolve_from_config_dir(tmp_path):
    config = load_config(write(tmp_path, "storage: {incidents_dir: out/v}"))
    assert config.storage.incidents_dir == (tmp_path / "out" / "v").resolve()


def test_overrides_win(tmp_path):
    config = load_config(write(tmp_path, "source: {uri: '0'}"), {"source.uri": "video.mp4", "receiver.enabled": False})
    assert config.source.uri == "video.mp4"
    assert config.receiver.enabled is False


def test_partial_timing_keeps_other_fields_and_rules(tmp_path):
    config = load_config(write(tmp_path, "timing: {zone: {enter_sec: 0.5}}"))
    assert config.timing[Rule.ZONE].enter_sec == 0.5
    assert config.timing[Rule.ZONE].exit_sec == 3.0  # умолчание именно для зоны
    assert config.timing[Rule.PPE].enter_sec == 3.0


def test_project_config_matches_requirements():
    t = load_config(DEFAULT_CONFIG).timings()[Rule.PPE]
    assert (t.enter_sec, t.exit_sec) == (3.0, 5.0)  # сообщить через 3 с, снять через 5 с


def test_unknown_key_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="ppe_conff"):
        load_config(write(tmp_path, "models: {ppe_conff: 0.3}"))


@pytest.mark.parametrize(
    "text",
    [
        "models: {ppe_conf: 1.5}",
        "models: {imgsz: 700}",
        "source: {process_every_n_frames: 0}",
        "rules: {ppe_regions: {helmet: hat}}",
        "calibration: {image_points: [[0, 0]]}",
        "timing: {ppe: {enter_ratio: 1.5}}",
        "timing: {ppe: {max_gap_sec: 0}}",
        "evidence: {clip_codec: h264x}",
    ],
)
def test_invalid_values(tmp_path, text):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))


def test_zones_roundtrip(tmp_path):
    path = tmp_path / "zones.yaml"
    zones = [Zone("crane", ((0.1, 0.1), (0.5, 0.1), (0.5, 0.6)))]
    save_zones(path, zones)
    assert load_zones(path) == zones


def test_zone_in_pixels_is_rejected(tmp_path):
    path = tmp_path / "zones.yaml"
    path.write_text("zones: [{name: a, polygon: [[10, 10], [200, 10], [200, 200]]}]", encoding="utf-8")
    with pytest.raises(ConfigError, match="долях"):
        load_zones(path)
