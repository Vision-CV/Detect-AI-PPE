"""Весь конвейер без нейросетей: фейковый детектор → пакеты инцидентов."""

from datetime import datetime

import numpy as np

from ppe_monitor.config import EvidenceConfig
from ppe_monitor.domain.analyzer import Analyzer
from ppe_monitor.domain.episode_tracker import EpisodeTracker, Timing
from ppe_monitor.domain.models import Zone
from ppe_monitor.domain.policy import PPEPolicy
from ppe_monitor.pipeline.evidence import EvidenceCollector
from ppe_monitor.pipeline.runner import Pipeline
from ppe_monitor.pipeline.source import Frame

from factories import helmet_on, person, vest_on

TIMINGS = {"ppe": Timing(enter_sec=3, exit_sec=5), "zone": Timing(enter_sec=1, exit_sec=3)}
FPS = 10


class FakeDetector:
    def __init__(self, scenario):
        self.scenario = scenario  # время (с) -> (persons, ppe)
        self.n = 0

    def detect(self, image):
        result = self.scenario(self.n / FPS)
        self.n += 1
        return result


def frames(seconds):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    for i in range(round(seconds * FPS)):
        yield Frame(i, image, i / FPS, datetime(2026, 10, 20, 12, 0, 0), i * 1000 / FPS)


def make_pipeline(scenario, zones=(), **evidence):
    packages = []
    cfg = EvidenceConfig(**({"clip_pre_sec": 1, "clip_post_sec": 1, "clip_width": 160} | evidence))
    collector = EvidenceCollector(cfg, packages.append, source_name="test.mp4")
    pipeline = Pipeline(
        FakeDetector(scenario),
        Analyzer(PPEPolicy(), zones),
        EpisodeTracker("cam1", TIMINGS),
        collector,
    )
    return pipeline, collector, packages


WORKER = person(1, (100, 20, 160, 220))


def no_helmet_then_helmet(t):
    """0–6 с без каски, потом в каске."""
    items = [vest_on(WORKER)] + ([helmet_on(WORKER)] if t >= 6 else [])
    return [WORKER], items


def test_violation_produces_opened_and_closed_packages():
    pipeline, collector, packages = make_pipeline(no_helmet_then_helmet)
    stats = pipeline.run(frames(15))
    collector.close()

    assert (stats.opened, stats.closed) == (1, 1)
    opened, closed = packages
    assert opened.status == "opened" and closed.status == "closed"
    assert opened.event.episode_id == closed.event.episode_id
    assert opened.event.violation_type == "helmet"
    assert opened.event.event_ts == 3.0  # 3 с после начала
    assert closed.event.event_ts == 11.0  # каска надета в 6.0 + 5 с
    assert closed.manifest()["duration_sec"] == 6.0
    assert closed.manifest()["video_offsets_ms"] == {"started": 0, "event": 11000, "ended": 6000}

    assert {f.kind for f in opened.files} == {"frame_start", "raw_start"}
    assert {f.kind for f in closed.files} == {"frame_start", "raw_start", "frame_best", "frame_end", "clip"}


def test_equipped_worker_produces_nothing():
    pipeline, collector, packages = make_pipeline(lambda t: ([WORKER], [helmet_on(WORKER), vest_on(WORKER)]))
    pipeline.run(frames(10))
    collector.close()
    assert packages == []


def test_end_of_video_closes_open_episode():
    pipeline, collector, packages = make_pipeline(lambda t: ([WORKER], [vest_on(WORKER)]))
    pipeline.run(frames(5))
    collector.close()
    assert [(p.status, p.event.close_reason) for p in packages][-1][1].value == "stream_ended"
    assert "clip" in {f.kind for f in packages[-1].files}


def test_zone_and_ppe_are_independent_episodes():
    zone = Zone("left", ((0.0, 0.0), (0.6, 0.0), (0.6, 1.0), (0.0, 1.0)))
    pipeline, collector, packages = make_pipeline(lambda t: ([WORKER], [vest_on(WORKER)]), zones=[zone])
    pipeline.run(frames(4))
    collector.close()
    opened = [p.event.violation_type for p in packages if p.status == "opened"]
    assert opened == ["zone:left", "helmet"]


def test_max_frames():
    pipeline, collector, _ = make_pipeline(lambda t: ([], []))
    assert pipeline.run(frames(100), max_frames=5).frames == 5
    collector.close()
