# config/settings.py
"""Глобальные настройки проекта. Авто-оптимизация под железо."""

from src.utils.hardware import get_hardware_profile

HARDWARE_OPT = get_hardware_profile()

DEVICE: str = HARDWARE_OPT["device"]
HALF_PRECISION: bool = HARDWARE_OPT["half"]
IMG_SIZE: int = HARDWARE_OPT["imgsz"]
PROCESS_EVERY_N_FRAMES: int = HARDWARE_OPT["skip_frames"]
TORCH_THREADS: int = HARDWARE_OPT["torch_threads"]
DATA_WORKERS: int = HARDWARE_OPT["workers"]

POSE_MODEL: str = "models_weights/yolov8n-pose.pt"
PPE_MODEL: str = HARDWARE_OPT["PPE_MODEL"]
CONF_THRESHOLD: float = 0.5
IOU_THRESHOLD: float = 0.5
VIOLATION_CONFIRMATION_SEC: float = 3.0
VIOLATION_COOLDOWN_SEC: float = 60.0
TRACK_LOSS_GRACE_SEC: float = 2.0
OUTPUT_DIR: str = "data/violations"
LOG_FILE: str = "data/logs/violations.jsonl"
CAMERA_ID: str = "cam_entrance"
REQUIRED_PPE_CLASSES: set[str] = {"helmet", "vest"}

CALIB_IMAGE_PTS = None
CALIB_WORLD_PTS = None

API_INCIDENT_URL: str = "https://localhost:7136/api/metrics/send-incident"
API_ROOM_ID: int = 42526769
API_INCIDENT_TYPE: int
API_PRIORITY: int = 1
API_SSL_VERIFY: bool = False
