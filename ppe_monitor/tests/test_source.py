"""Источники кадров: видеофайл (порядок, время, пропуск) и поток (свежий кадр, переподключение)."""

import threading
import time

import numpy as np
import pytest

from ppe_monitor.config import SourceConfig
from ppe_monitor.pipeline import source as source_module
from ppe_monitor.pipeline.source import FileSource, LiveSource, SourceError, _is_live, open_source

from conftest import frame_number, write_video

# ------------------------------------------------------------------ файл


def test_file_yields_every_frame_with_video_time(synthetic_video):
    src = FileSource(synthetic_video, every_n=1, resize_width=None)
    frames = list(src)
    src.close()
    assert [f.index for f in frames] == list(range(30))
    assert [frame_number(f.image) for f in frames] == list(range(30))
    assert frames[10].timestamp == pytest.approx(1.0, abs=0.01)  # 10 к/с
    assert frames[10].video_pos_ms == pytest.approx(1000, abs=10)


def test_file_every_n_skips_but_keeps_true_time(synthetic_video):
    src = FileSource(synthetic_video, every_n=3, resize_width=None)
    frames = list(src)
    src.close()
    assert [f.index for f in frames] == list(range(0, 30, 3))
    assert [frame_number(f.image) for f in frames] == list(range(0, 30, 3))  # нужный кадр, не соседний
    assert [round(f.timestamp, 1) for f in frames] == [i / 10 for i in range(0, 30, 3)]


def test_file_resize_keeps_aspect(synthetic_video):
    src = FileSource(synthetic_video, every_n=1, resize_width=80)
    first = next(iter(src))
    src.close()
    assert first.size == (80, 60)


def test_missing_file_is_clear_error(tmp_path):
    with pytest.raises(SourceError, match="не найден"):
        FileSource(tmp_path / "nope.mp4", 1, None)


def test_broken_file_is_clear_error(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(SourceError):
        FileSource(bad, 1, None)


@pytest.mark.parametrize(
    "uri,live",
    [("0", True), ("2", True), ("rtsp://cam/1", True), ("HTTP://x/stream", True),
     ("video.mp4", False), (r"C:\videos\a.mp4", False)],
)
def test_live_or_file_detection(uri, live):
    assert _is_live(uri) is live


def test_open_source_picks_file(synthetic_video):
    src = open_source(SourceConfig(uri=str(synthetic_video)), every_n=1)
    assert isinstance(src, FileSource)
    src.close()


# ------------------------------------------------------------------ поток


class FakeCapture:
    """Камера: выдаёт заданные кадры (значение = номер), затем «обрыв»."""

    def __init__(self, values, delay=0.0):
        self.values = list(values)
        self.delay = delay
        self.released = False

    def isOpened(self):  # noqa: N802 — имя как у cv2.VideoCapture
        return True

    def read(self):
        if self.delay:
            time.sleep(self.delay)
        if not self.values:
            time.sleep(0.001)
            return False, None
        v = self.values.pop(0)
        return True, np.full((4, 4, 3), v, np.uint8)

    def release(self):
        self.released = True


def fake_cameras(monkeypatch, *captures):
    queue = list(captures)
    opened = []

    def open_capture(uri):
        cap = queue.pop(0) if queue else FakeCapture([])
        opened.append(cap)
        return cap

    monkeypatch.setattr(source_module, "_open_capture", open_capture)
    return opened


def collect(src, timeout=3.0, limit=None, per_frame_sleep=0.0):
    values, deadline = [], time.monotonic() + timeout
    for frame in src:
        values.append(int(frame.image[0, 0, 0]))
        if per_frame_sleep:
            time.sleep(per_frame_sleep)
        if (limit and len(values) >= limit) or time.monotonic() > deadline:
            break
    return values


def test_live_reconnects_after_drop(monkeypatch):
    opened = fake_cameras(monkeypatch, FakeCapture([1, 2, 3], delay=0.01), FakeCapture([4, 5, 6], delay=0.01))
    src = LiveSource("rtsp://cam", None, reconnect_attempts=0, reconnect_delay=0.01)
    values = collect(src, limit=6)
    src.close()
    assert values[-1] == 6  # кадры второй «сессии» получены
    assert opened[0].released


def test_live_stops_after_reconnect_attempts(monkeypatch):
    """Текущее поведение: после N неудачных попыток источник завершается."""
    fake_cameras(monkeypatch, FakeCapture([1, 2], delay=0.01))
    src = LiveSource("rtsp://cam", None, reconnect_attempts=2, reconnect_delay=0.01)
    done = threading.Event()
    values = []

    def consume():
        values.extend(collect(src, timeout=5))
        done.set()

    threading.Thread(target=consume, daemon=True).start()
    assert done.wait(5), "источник не завершился после исчерпания попыток"
    src.close()
    assert values == [1, 2]


def test_live_gives_latest_frame_when_consumer_is_slow(monkeypatch):
    fake_cameras(monkeypatch, FakeCapture(list(range(100)), delay=0.002))
    src = LiveSource("0", None, reconnect_attempts=1, reconnect_delay=0.01)
    values = collect(src, timeout=5, per_frame_sleep=0.05)
    src.close()
    assert values == sorted(set(values))  # без повторов, по возрастанию
    assert len(values) < 50  # медленный потребитель пропускает старые кадры
    assert values[-1] == 99  # но получает самый свежий


def test_live_timestamps_are_monotonic(monkeypatch):
    fake_cameras(monkeypatch, FakeCapture(list(range(20)), delay=0.005))
    src = LiveSource("0", None, reconnect_attempts=1, reconnect_delay=0.01)
    stamps = []
    for frame in src:
        stamps.append(frame.timestamp)
        if len(stamps) == 5:
            break
    src.close()
    assert stamps == sorted(stamps) and stamps[-1] > stamps[0]


def test_live_unavailable_camera_is_error(monkeypatch):
    class Closed(FakeCapture):
        def isOpened(self):  # noqa: N802
            return False

    fake_cameras(monkeypatch, Closed([]))
    with pytest.raises(SourceError):
        LiveSource("rtsp://cam", None, 1, 0.01)


def test_write_video_helper_roundtrip(tmp_path):
    path = write_video(tmp_path / "v.avi", 5)
    src = FileSource(path, 1, None)
    assert [frame_number(f.image) for f in src] == [0, 1, 2, 3, 4]
    src.close()
