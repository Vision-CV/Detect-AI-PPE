# src/core/hysteresis.py
from collections import deque
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ViolationState(Enum):
    NORMAL = "normal"
    UNKNOWN = "unknown"
    VIOLATION = "violation"


class HysteresisTracker:
    def __init__(
            self, window_size: int = 30,
            enter_thresh: float = 0.7,
            exit_thresh: float = 0.5):

        self.window_size = window_size
        self.enter_thresh = enter_thresh
        self.exit_thresh = exit_thresh
        self.history: deque[bool] = deque(maxlen=window_size)
        self.state: ViolationState = ViolationState.NORMAL

    def update(self, is_violating: bool) -> ViolationState:
        self.history.append(is_violating)

        if len(self.history) < self.window_size * 0.3:
            return ViolationState.UNKNOWN

        ratio = sum(self.history) / len(self.history)

        if self.state == ViolationState.NORMAL:
            if ratio >= self.enter_thresh:
                self.state = ViolationState.VIOLATION

        elif self.state == ViolationState.VIOLATION:
            if ratio <= self.exit_thresh:
                self.state = ViolationState.NORMAL

        return self.state

    def should_alert(self) -> bool:
        return self.state == ViolationState.VIOLATION
