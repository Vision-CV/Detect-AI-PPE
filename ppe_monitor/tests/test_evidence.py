"""Сборщик доказательств: состав пакетов, клип, размытие, хеши."""

import hashlib
from datetime import datetime

import cv2
import numpy as np

from ppe_monitor.config import EvidenceConfig
from ppe_monitor.domain.analyzer import Analyzer
from ppe_monitor.domain.models import CloseReason, EpisodeEvent, EpisodeKind, Rule
from ppe_monitor.domain.policy import PPEPolicy
from ppe_monitor.pipeline.evidence import EvidenceCollector, blur_regions
from ppe_monitor.pipeline.source import Frame

from factories import person, vest_on

WORKER = person(1, (100, 20, 160, 220))
ANALYSIS = Analyzer(PPEPolicy()).analyze([WORKER], [vest_on(WORKER)], (320, 240))
START = datetime(2026, 10, 20, 12, 0, 0)


def frame(i, fps=10):
    img = np.full((240, 320, 3), 40, np.uint8)
    cv2.circle(img, (130, 40), 15, (200, 180, 160), -1)  # «голова»
    return Frame(i, img, i / fps, START, None)


def event(kind, ts, **kw):
    return EpisodeEvent(
        kind=kind, episode_id="ep1", camera_id="cam1", rule=Rule.PPE, subject="helmet",
        track_id=1, track_ids=(1,), started_ts=0.0, event_ts=ts, box=WORKER.box,
        confidence=0.9, violating_sec=ts, **kw,
    )


def collect(config, steps):
    """steps: [(номер_кадра, [события])] → опубликованные пакеты."""
    packages = []
    c = EvidenceCollector(config, packages.append, source_name="cam")
    for i, events in steps:
        f = frame(i)
        c.on_frame(f, ANALYSIS, events, render=lambda f=f: f.image.copy(), open_episodes={"ep1": (1, Rule.PPE, "helmet")})
    c.close()
    return packages


CFG = EvidenceConfig(clip_pre_sec=1, clip_post_sec=1, clip_width=160)


def test_opened_is_published_immediately_without_clip():
    [opened] = collect(CFG, [(i, []) for i in range(30)] + [(30, [event(EpisodeKind.OPENED, 3.0)])])
    assert opened.is_alert
    assert [f.name for f in opened.files] == ["frame_start.jpg", "raw_start.jpg"]


def test_closed_contains_clip_and_valid_hashes():
    steps = [(i, []) for i in range(30)] + [(30, [event(EpisodeKind.OPENED, 3.0)])]
    steps += [(i, []) for i in range(31, 60)]
    steps += [(60, [event(EpisodeKind.CLOSED, 6.0, ended_ts=5.0, close_reason=CloseReason.RESOLVED)])]
    opened, closed = collect(CFG, steps)

    names = [f.name for f in closed.files]
    assert names == ["frame_start.jpg", "raw_start.jpg", "frame_best.jpg", "frame_end.jpg", "clip.mp4"]
    for item in closed.manifest()["evidence"]:
        data = next(f.data for f in closed.files if f.name == item["name"])
        assert hashlib.sha256(data).hexdigest() == item["sha256"]

    clip = next(f.data for f in closed.files if f.name == "clip.mp4")
    assert len(clip) > 1000


def test_clip_spans_pre_and_post_seconds(tmp_path):
    steps = [(i, []) for i in range(30)] + [(30, [event(EpisodeKind.OPENED, 3.0)])]
    steps += [(i, []) for i in range(31, 60)]
    steps += [(60, [event(EpisodeKind.CLOSED, 6.0, ended_ts=5.0, close_reason=CloseReason.RESOLVED)])]
    _, closed = collect(CFG, steps)
    path = tmp_path / "clip.mp4"
    path.write_bytes(next(f.data for f in closed.files if f.name == "clip.mp4"))
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert 19 <= n <= 22  # ±1 с при 10 к/с


def test_track_lost_has_no_end_frame():
    steps = [(0, [event(EpisodeKind.OPENED, 0.0)]), (1, [event(EpisodeKind.CLOSED, 0.1, ended_ts=0.0,
                                                                   close_reason=CloseReason.TRACK_LOST)])]
    _, closed = collect(CFG, steps)
    assert "frame_end" not in {f.kind for f in closed.files}


def test_clip_can_be_disabled():
    cfg = EvidenceConfig(clip_enabled=False)
    steps = [(0, [event(EpisodeKind.OPENED, 0.0)]), (1, [event(EpisodeKind.CLOSED, 0.1, ended_ts=0.0,
                                                                   close_reason=CloseReason.RESOLVED)])]
    _, closed = collect(cfg, steps)
    assert "clip" not in {f.kind for f in closed.files}


def test_blur_faces_keeps_raw_frame_local_only():
    cfg = EvidenceConfig(clip_enabled=False, blur_faces=True)
    [opened] = collect(cfg, [(0, [event(EpisodeKind.OPENED, 0.0)])])
    raw = next(f for f in opened.files if f.kind == "raw_start")
    assert raw.local_only
    assert "raw_start.jpg" not in {e["name"] for e in opened.manifest()["evidence"]}
    assert "raw_start.jpg" in {e["name"] for e in opened.manifest(include_local=True)["evidence"]}


def test_blur_regions_changes_only_region():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    out = blur_regions(img, [(10, 10, 40, 40)])
    assert not np.array_equal(out[10:40, 10:40], img[10:40, 10:40])
    assert np.array_equal(out[50:, 50:], img[50:, 50:])


def test_wall_times_derived_from_frame_clock():
    # Кадр 30 (t=3.0 с) снят в 12:00:00; нарушение началось при t=0.0 → за 3 с до этого.
    [opened] = collect(CFG, [(30, [event(EpisodeKind.OPENED, 3.0)])])
    m = opened.manifest()
    assert m["event_at"] == "2026-10-20T12:00:00.000"
    assert m["started_at"] == "2026-10-20T11:59:57.000"
