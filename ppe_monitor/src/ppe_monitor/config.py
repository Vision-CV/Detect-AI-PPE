"""Конфигурация: YAML-файл + переопределения из командной строки.

Конфиг — это просто данные. Здесь нет определения железа, импорта torch
и прочих побочных эффектов: модуль можно импортировать где угодно.

Относительные пути в конфиге считаются от папки, где лежит сам файл конфига,
а не от текущей директории — поэтому запуск из любого места работает одинаково.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from ppe_monitor.domain.models import Rule, Zone
from ppe_monitor.domain.policy import PPEPolicy, Region
from ppe_monitor.domain.episode_tracker import Timing

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

AUTO = "auto"

# Пометка для полей-путей: они резолвятся относительно файла конфига.
_PATH = {"path": True}


class ConfigError(ValueError):
    """Ошибка в конфигурации. Сообщение должно объяснять, что исправить."""


@dataclass(frozen=True)
class SourceConfig:
    uri: str = "0"  # индекс камеры, путь к файлу или rtsp/http-адрес
    resize_width: int | None = None
    process_every_n_frames: int | str = AUTO  # только для файлов
    reconnect_attempts: int = 5  # для потоков; 0 — пытаться бесконечно
    reconnect_delay_sec: float = 2.0


@dataclass(frozen=True)
class HardwareConfig:
    device: str = AUTO  # auto | cpu | cuda | cuda:0 | mps
    half: bool | str = AUTO
    torch_threads: int | str = AUTO


@dataclass(frozen=True)
class ModelsConfig:
    pose: Path = field(default=Path("models_weights/yolov8n-pose.pt"), metadata=_PATH)
    ppe: Path = field(default=Path("models_weights/PPE_M_YOLO.pt"), metadata=_PATH)
    imgsz: int | str = AUTO
    person_conf: float = 0.5
    ppe_conf: float = 0.35
    iou: float = 0.5
    tracker: str = "bytetrack.yaml"
    ignore_classes: tuple[str, ...] = ("human", "person")


@dataclass(frozen=True)
class RulesConfig:
    required_ppe: tuple[str, ...] = ("helmet", "vest")
    ppe_regions: dict[str, str] = field(
        default_factory=lambda: {"helmet": Region.HEAD, "vest": Region.TORSO}
    )
    max_width_ratio: dict[str, float] = field(
        default_factory=lambda: {"helmet": 1.2, "vest": 1.6}
    )
    max_per_person: dict[str, int] = field(default_factory=lambda: {"gloves": 2, "boots": 2})
    keypoint_min_visibility: float = 0.3
    edge_margin_px: float = 4.0
    zones_file: Path | None = field(default=None, metadata=_PATH)

    def to_policy(self) -> PPEPolicy:
        return PPEPolicy(
            required=frozenset(self.required_ppe),
            regions=dict(self.ppe_regions),
            max_width_ratio=dict(self.max_width_ratio),
            max_per_person=dict(self.max_per_person),
            min_keypoint_visibility=self.keypoint_min_visibility,
            edge_margin_px=self.edge_margin_px,
        )


@dataclass(frozen=True)
class TimingConfig:
    """Гистерезис для одного правила (см. domain/episode_tracker.py)."""

    enter_sec: float = 3.0  # нарушение видно столько секунд → эпизод открыт
    exit_sec: float = 5.0  # порядок виден столько секунд подряд → эпизод закрыт
    enter_ratio: float = 0.6  # доля «нарушение» в окне enter_sec (устойчивость к миганию)
    lost_ttl_sec: float = 30.0  # сколько помнить пропавшего человека
    reminder_sec: float = 0.0  # повторное оповещение, пока длится (0 — выкл.)
    max_gap_sec: float = 2.0  # промежуток между кадрами длиннее — «не видно»

    def to_timing(self) -> Timing:
        return Timing(**asdict(self))


def _default_timing() -> dict[str, TimingConfig]:
    return {
        Rule.PPE: TimingConfig(enter_sec=3.0, exit_sec=5.0),
        Rule.ZONE: TimingConfig(enter_sec=1.0, exit_sec=3.0),
    }


@dataclass(frozen=True)
class TrackingConfig:
    # Новый трек ближе этого расстояния (в ростах человека) к пропавшему
    # нарушителю наследует его эпизод (защита от дублей при ID switch).
    reid_max_distance: float = 1.0


@dataclass(frozen=True)
class EvidenceConfig:
    """Доказательства в пакете инцидента."""

    jpeg_quality: int = 90
    clip_enabled: bool = True
    clip_pre_sec: float = 5.0  # секунд до подтверждения нарушения
    clip_post_sec: float = 5.0  # секунд после
    clip_width: int = 960  # ширина кадров клипа (экономия памяти и трафика)
    clip_codec: str = "mp4v"  # avc1 (H.264) играется в браузере, но нужен OpenH264
    blur_faces: bool = False  # размывать головы на отправляемых кадрах и клипе


@dataclass(frozen=True)
class StorageConfig:
    """Локальная копия пакетов и журнал событий."""

    enabled: bool = True
    incidents_dir: Path = field(default=Path("data/incidents"), metadata=_PATH)
    journal_file: Path = field(default=Path("data/logs/incidents.jsonl"), metadata=_PATH)


@dataclass(frozen=True)
class ReceiverConfig:
    """Сервер-приёмник пакетов инцидентов (см. src/ppe_receiver)."""

    enabled: bool = True
    url: str = "http://localhost:8000/api/v1/incidents"
    ssl_verify: bool = True
    timeout_sec: float = 30.0
    retry_max_delay_sec: float = 60.0
    shutdown_drain_sec: float = 15.0  # при выходе досылать очередь не дольше этого
    outbox_dir: Path = field(default=Path("data/outbox/receiver"), metadata=_PATH)


@dataclass(frozen=True)
class VisDashConfig:
    """Старый дашборд VisDash: только оповещение (кадр + 4 поля) при открытии эпизода."""

    enabled: bool = False
    url: str = "http://localhost:5000/api/Metrics/send-incident"
    cam_id: int | None = None  # None — взять цифры из camera_id
    room_id: int = 0
    priority: int = 1
    ssl_verify: bool = False
    timeout_sec: float = 10.0
    retry_max_delay_sec: float = 60.0
    shutdown_drain_sec: float = 15.0
    outbox_dir: Path = field(default=Path("data/outbox/visdash"), metadata=_PATH)
    # violation_type ("helmet", "zone:crane") или правило ("ppe", "zone") или
    # "default" -> числовой Type в API дашборда.
    incident_types: dict[str, int] = field(default_factory=lambda: {"default": 0, "zone": 2})


@dataclass(frozen=True)
class DisplayConfig:
    enabled: bool = True
    window_name: str = "PPE Monitoring"


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    file: Path | None = field(default=Path("data/logs/ppe_monitor.log"), metadata=_PATH)


@dataclass(frozen=True)
class CalibrationConfig:
    """4 точки на кадре (пиксели) и соответствующие им точки пола (метры)."""

    image_points: tuple[tuple[float, float], ...] | None = None
    world_points: tuple[tuple[float, float], ...] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.image_points and self.world_points)


@dataclass(frozen=True)
class AppConfig:
    camera_id: str = "cam_01"
    source: SourceConfig = field(default_factory=SourceConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    timing: dict[str, TimingConfig] = field(default_factory=_default_timing)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    receiver: ReceiverConfig = field(default_factory=ReceiverConfig)
    visdash: VisDashConfig = field(default_factory=VisDashConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def timings(self) -> dict[str, Timing]:
        return {rule: t.to_timing() for rule, t in self.timing.items()}


# ---------------------------------------------------------------- загрузка


def load_config(path: Path | str | None = None, overrides: Mapping[str, Any] | None = None) -> AppConfig:
    """Читает YAML, применяет переопределения вида {"source.uri": "x.mp4"} и проверяет."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_file():
        raise ConfigError(f"Файл конфигурации не найден: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: корень конфига должен быть словарём")

    for dotted, value in (overrides or {}).items():
        _set_dotted(data, dotted, value)

    config = _build(AppConfig, data, path.resolve().parent, "config")
    _validate(config)
    return config


def _set_dotted(data: dict, dotted: str, value: Any) -> None:
    *parents, last = dotted.split(".")
    node = data
    for key in parents:
        node = node.setdefault(key, {})
    node[last] = value


def _build(cls: type, data: Any, base_dir: Path, where: str) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: ожидался словарь, получено {type(data).__name__}")

    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"{where}: неизвестные ключи {sorted(unknown)}; допустимые: {sorted(known)}")

    hints = get_type_hints(cls)
    kwargs = {}
    for name, value in data.items():
        f = known[name]
        hint = hints[name]
        if is_dataclass(hint):
            value = _build(hint, value, base_dir, f"{where}.{name}")
        elif cls is AppConfig and name == "timing":
            value = _build_timing(value, base_dir, f"{where}.timing")
        elif f.metadata.get("path") and value is not None:
            value = _resolve_path(value, base_dir)
        elif isinstance(value, list):
            value = _to_tuple(value)
        kwargs[name] = value
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ConfigError(f"{where}: {e}") from e


def _build_timing(data: Any, base_dir: Path, where: str) -> dict[str, TimingConfig]:
    result = _default_timing()
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: ожидался словарь правило -> тайминги")
    for rule, value in data.items():
        # Частичное переопределение: недостающие поля — из умолчаний этого правила.
        base = asdict(result[rule]) if rule in result else {}
        merged = base | (value or {}) if isinstance(value, dict) else value
        result[rule] = _build(TimingConfig, merged, base_dir, f"{where}.{rule}")
    return result


def _resolve_path(value: Any, base_dir: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p).resolve()


def _to_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_to_tuple(v) for v in value)
    return value


def _validate(c: AppConfig) -> None:
    errors = []
    for name in ("person_conf", "ppe_conf", "iou"):
        v = getattr(c.models, name)
        if not 0 < v <= 1:
            errors.append(f"models.{name} должен быть в (0, 1], сейчас {v}")
    n = c.source.process_every_n_frames
    if n != AUTO and (not isinstance(n, int) or n < 1):
        errors.append(f"source.process_every_n_frames: целое >= 1 или 'auto', сейчас {n!r}")
    imgsz = c.models.imgsz
    if imgsz != AUTO and (not isinstance(imgsz, int) or imgsz % 32):
        errors.append(f"models.imgsz: кратное 32 или 'auto', сейчас {imgsz!r}")
    for cls, region in c.rules.ppe_regions.items():
        if region not in Region.ALL:
            errors.append(f"rules.ppe_regions.{cls}: '{region}', допустимо {Region.ALL}")
    for rule in (Rule.PPE, Rule.ZONE):
        if rule not in c.timing:
            errors.append(f"timing.{rule}: не задан")
    for rule, t in c.timing.items():
        if t.enter_sec <= 0 or t.exit_sec <= 0 or t.max_gap_sec <= 0:
            errors.append(f"timing.{rule}: enter_sec, exit_sec и max_gap_sec должны быть > 0")
        if not 0 < t.enter_ratio <= 1:
            errors.append(f"timing.{rule}.enter_ratio должен быть в (0, 1], сейчас {t.enter_ratio}")
        if t.lost_ttl_sec < t.max_gap_sec:
            errors.append(f"timing.{rule}.lost_ttl_sec должен быть не меньше max_gap_sec")
    ev = c.evidence
    if ev.clip_pre_sec < 0 or ev.clip_post_sec < 0 or ev.clip_width < 64:
        errors.append("evidence: clip_pre_sec/clip_post_sec >= 0, clip_width >= 64")
    if len(ev.clip_codec) != 4:
        errors.append(f"evidence.clip_codec: 4 символа (FourCC), сейчас {ev.clip_codec!r}")
    cal = c.calibration
    if bool(cal.image_points) != bool(cal.world_points):
        errors.append("calibration: задайте и image_points, и world_points (или ни то, ни другое)")
    elif cal.enabled and (len(cal.image_points) != 4 or len(cal.world_points) != 4):
        errors.append("calibration: нужно ровно 4 пары точек")
    if errors:
        raise ConfigError("Ошибки в конфигурации:\n  - " + "\n  - ".join(errors))


# ---------------------------------------------------------------- зоны


def load_zones(path: Path) -> list[Zone]:
    """Файл зон: {zones: [{name: str, polygon: [[x, y], ...]}]}, x/y в долях кадра."""
    if not path.is_file():
        raise ConfigError(f"Файл зон не найден: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    zones = []
    for i, raw in enumerate(data.get("zones") or []):
        where = f"{path.name}: zones[{i}]"
        name = raw.get("name")
        polygon = raw.get("polygon") or []
        if not name:
            raise ConfigError(f"{where}: не задано имя")
        if len(polygon) < 3:
            raise ConfigError(f"{where} '{name}': нужно минимум 3 точки")
        points = tuple((float(x), float(y)) for x, y in polygon)
        if any(not (0 <= v <= 1) for p in points for v in p):
            raise ConfigError(f"{where} '{name}': координаты должны быть в долях кадра (0..1)")
        zones.append(Zone(str(name), points))
    return zones


def save_zones(path: Path, zones: list[Zone]) -> None:
    data = {
        "zones": [
            {"name": z.name, "polygon": [[round(x, 4), round(y, 4)] for x, y in z.polygon]}
            for z in zones
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
