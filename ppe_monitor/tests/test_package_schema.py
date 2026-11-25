"""Референс пакета (docs/) не должен расходиться с кодом.

Манифесты, которые реально создаёт конвейер во всех сценариях, и примеры из
документации проверяются по JSON Schema docs/incident-package.schema.json.
"""

import json
from datetime import datetime

import numpy as np
import pytest

from ppe_monitor.config import PROJECT_ROOT, EvidenceConfig
from ppe_monitor.domain.analyzer import Analyzer
from ppe_monitor.domain.episode_tracker import EpisodeTracker, Timing
from ppe_monitor.domain.models import Zone
from ppe_monitor.domain.policy import PPEPolicy
from ppe_monitor.pipeline.evidence import EvidenceCollector
from ppe_monitor.pipeline.runner import Pipeline
from ppe_monitor.pipeline.source import Frame
from ppe_monitor.sinks.base import SCHEMA

from factories import helmet_on, person, vest_on

jsonschema = pytest.importorskip("jsonschema")

DOCS = PROJECT_ROOT / "docs"
SCHEMA_DOC = json.loads((DOCS / "incident-package.schema.json").read_text(encoding="utf-8"))
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA_DOC, format_checker=jsonschema.FormatChecker())
WORKER = person(1, (100, 20, 160, 220))


def errors(manifest):
    return [f"{'/'.join(map(str, e.path))}: {e.message}" for e in VALIDATOR.iter_errors(manifest)]


def run(scenario, seconds, *, zones=(), live=False, reminder=0.0, **evidence):
    packages = []
    collector = EvidenceCollector(
        EvidenceConfig(**({"clip_pre_sec": 1, "clip_post_sec": 1, "clip_width": 160} | evidence)),
        packages.append, source_name="rtsp://cam" if live else "test.mp4",
        model_info={"pose": "pose.pt", "ppe": "ppe.pt", "imgsz": 640},
        locate=lambda p, size: (1.5, 4.25),
    )
    timings = {"ppe": Timing(enter_sec=3, exit_sec=5, reminder_sec=reminder, lost_ttl_sec=2),
               "zone": Timing(enter_sec=1, exit_sec=3, lost_ttl_sec=2)}

    class Detector:
        n = 0

        def detect(self, image):
            Detector.n += 1
            return scenario((Detector.n - 1) / 10)

    pipeline = Pipeline(Detector(), Analyzer(PPEPolicy(), zones), EpisodeTracker("cam1", timings), collector)
    image = np.zeros((240, 320, 3), np.uint8)
    frames = (
        Frame(i, image, i / 10, datetime(2026, 10, 20, 12, 0, 0), None if live else i * 100)
        for i in range(int(seconds * 10))
    )
    pipeline.run(frames)
    collector.close()
    return packages


SCENARIOS = {
    "resolved": lambda: run(lambda t: ([WORKER], [vest_on(WORKER)] + ([helmet_on(WORKER)] if t >= 5 else [])), 12),
    "stream_ended": lambda: run(lambda t: ([WORKER], [vest_on(WORKER)]), 5),
    "track_lost": lambda: run(lambda t: ([WORKER], [vest_on(WORKER)]) if t < 4 else ([], []), 8),
    "reminder": lambda: run(lambda t: ([WORKER], [vest_on(WORKER)]), 6, reminder=1.0),
    "zone": lambda: run(lambda t: ([WORKER], [helmet_on(WORKER), vest_on(WORKER)]), 3,
                        zones=[Zone("crane", ((0, 0), (1, 0), (1, 1), (0, 1)))]),
    "live_source": lambda: run(lambda t: ([WORKER], [vest_on(WORKER)]), 5, live=True),
    "blur_faces": lambda: run(lambda t: ([WORKER], [vest_on(WORKER)]), 5, blur_faces=True),
    "no_clip": lambda: run(lambda t: ([WORKER], [vest_on(WORKER)]), 5, clip_enabled=False),
}


def test_schema_itself_is_valid():
    jsonschema.Draft202012Validator.check_schema(SCHEMA_DOC)
    assert SCHEMA_DOC["properties"]["schema"]["const"] == SCHEMA  # версия схемы = версия в коде


@pytest.mark.parametrize("name", SCENARIOS)
def test_generated_manifests_match_schema(name):
    packages = SCENARIOS[name]()
    assert packages, f"сценарий {name} не дал пакетов"
    for package in packages:
        for manifest in (package.manifest(), package.manifest(include_local=True)):
            assert errors(manifest) == [], f"{name}/{package.status}"


def test_scenarios_cover_all_statuses_reasons_and_kinds():
    """Если в коде появится новый статус/причина/вид файла — этот тест напомнит обновить референс."""
    seen_status, seen_reason, seen_kind = set(), set(), set()
    for make in SCENARIOS.values():
        for p in make():
            m = p.manifest(include_local=True)
            seen_status.add(m["status"])
            seen_reason.add(m["close_reason"])
            seen_kind.update(e["kind"] for e in m["evidence"])
    props = SCHEMA_DOC["properties"]
    assert seen_status == set(props["status"]["enum"])
    assert seen_reason == set(props["close_reason"]["enum"])
    assert seen_kind == set(SCHEMA_DOC["$defs"]["evidence"]["properties"]["kind"]["enum"])


@pytest.mark.parametrize("example", ["opened.json", "closed.json"])
def test_documentation_examples_are_valid(example):
    manifest = json.loads((DOCS / "examples" / example).read_text(encoding="utf-8"))
    assert errors(manifest) == []


def test_reference_mentions_every_field():
    reference = (DOCS / "incident-package.md").read_text(encoding="utf-8")
    missing = [f for f in SCHEMA_DOC["required"] if f"`{f}`" not in reference]
    assert missing == [], f"поля не описаны в docs/incident-package.md: {missing}"


@pytest.mark.parametrize(
    "patch",
    [
        {"status": "closed"},  # closed без ended_at / close_reason
        {"close_reason": "resolved"},  # причина у открытого эпизода
        {"rule": "zone"},  # violation_type без префикса zone:
        {"evidence": [{"name": "a.jpg", "kind": "frame_start", "content_type": "image/jpeg", "size": 1, "sha256": "xyz"}]},
        {"started_at": "20.10.2026 12:00"},
        {"person_box": [1, 2, 3]},
    ],
)
def test_schema_rejects_broken_manifests(patch):
    manifest = json.loads((DOCS / "examples" / "opened.json").read_text(encoding="utf-8")) | patch
    assert errors(manifest) != []
