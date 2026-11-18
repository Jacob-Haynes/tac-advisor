"""Grid references against published reference points, and their use in the LLM context."""

import time

import pytest

from advisor import grid
from advisor.context_builder import SYSTEM_PROMPT, build_context
from advisor.cot_listener import DroneInfo, PositionInfo


def dms(d, m, s):
    return d + m / 60 + s / 3600


def test_utm_cn_tower():
    # Wikipedia "Universal Transverse Mercator": CN Tower 43°38′33.24″N 79°23′13.7″W = 17T 630084 4833438
    zone, band, e, n = grid.latlon_to_utm(dms(43, 38, 33.24), -dms(79, 23, 13.7))
    assert (zone, band, int(e), int(n)) == (17, "T", 630084, 4833438)
    assert grid.to_mgrs(dms(43, 38, 33.24), -dms(79, 23, 13.7)) == "17T PJ 30084 33438"


def test_mgrs_eiffel_tower_and_precision():
    assert grid.to_mgrs(48.8582, 2.2945) == "31U DQ 48251 11932"
    assert grid.to_mgrs(48.8582, 2.2945, digits=3) == "31U DQ 482 119"


def test_mgrs_salisbury_plain_southern_hemisphere_and_norway():
    assert grid.to_mgrs(51.15, -1.75) == "30U WB 87427 67248"
    assert grid.to_mgrs(-33.8568, 151.2153).startswith("56H LH ")
    assert grid.to_mgrs(60.0, 5.0).startswith("32V ")  # Norway exception (would be 31V)


def test_osgb_worked_example():
    # OS "A guide to coordinate systems in Great Britain", annex C: Caister water tower (OSGB36)
    e, n = grid.osgb36_to_en(dms(52, 39, 27.2531), dms(1, 43, 4.5177))
    assert e == pytest.approx(651409.903, abs=0.01) and n == pytest.approx(313177.270, abs=0.01)
    assert grid.en_to_bng(e, n) == "TG 51409 13177"


def test_bng_from_wgs84():
    assert grid.to_bng(51.5007, -0.1246).startswith("TQ 302")  # Westminster
    assert grid.to_bng(55.9533, -3.1883).startswith("NT 25")  # Edinburgh
    assert grid.to_bng(51.15, -1.75).startswith("SU 1")  # Salisbury Plain
    assert grid.to_bng(48.8582, 2.2945) is None  # Paris: off the grid


def test_grid_ref_formats_and_fallbacks(monkeypatch):
    monkeypatch.setenv("GRID_FORMAT", "bng")
    assert grid.grid_ref(51.15, -1.75).startswith("SU ")
    assert grid.grid_ref(48.8582, 2.2945) == "31U DQ 48251 11932"  # outside GB -> MGRS
    monkeypatch.setenv("GRID_FORMAT", "latlon")
    assert grid.grid_ref(51.15, -1.75) == "51.15000 -1.75000"
    monkeypatch.setenv("GRID_FORMAT", "nonsense")
    assert grid.grid_ref(51.15, -1.75) == "30U WB 87427 67248"
    assert grid.grid_ref(0.0, 0.0) == "no position" and grid.grid_ref(None, None) == "no position"
    assert grid.grid_ref(85.0, 0.0) == "85.00000 0.00000"  # polar: no UPS, plain lat/lon


def test_relative_direction_and_distance():
    assert grid.relative(51.16, -1.76, 51.15, -1.75, "HANGOVER") == "1.3 km NW of HANGOVER"
    assert grid.relative(51.15, -1.7486, 51.15, -1.75, "HANGOVER") == "98 m E of HANGOVER"
    assert grid.relative(51.15, -1.75, 51.15, -1.75, "HANGOVER") == "at HANGOVER"


def test_context_uses_grid_refs_not_lat_lon(monkeypatch):
    monkeypatch.delenv("GRID_FORMAT", raising=False)
    monkeypatch.delenv("SENSOR_LAT", raising=False)
    now = time.time()
    state = {
        "positions": {"H": PositionInfo(callsign="HANGOVER", lat=51.15, lon=-1.75, timestamp=now)},
        "drones": [DroneInfo(uid="D1", lat=51.16, lon=-1.76, alt=40.0, callsign="UAV 1", timestamp=now)],
    }
    ctx = build_context(state)
    assert "GRID: MGRS." in ctx
    assert "HANGOVER: grid=30U WB 87427 67248" in ctx
    assert "UAV 1: grid=" + grid.to_mgrs(51.16, -1.76) + " (1.3 km NW of HANGOVER)" in ctx
    assert "lat=" not in ctx and "51.15" not in ctx  # nothing for the model to dress up as a grid
    assert "never" in SYSTEM_PROMPT and "invent" in SYSTEM_PROMPT


def test_context_relative_to_op_without_units(monkeypatch):
    monkeypatch.setenv("SENSOR_LAT", "51.15")
    monkeypatch.setenv("SENSOR_LON", "-1.75")
    monkeypatch.setenv("GRID_FORMAT", "bng")
    now = time.time()
    ctx = build_context({"drones": [DroneInfo(uid="D1", lat=51.16, lon=-1.76, alt=40.0, timestamp=now)]})
    assert "GRID: BNG" in ctx and "(1.3 km NW of the OP)" in ctx and "grid=SU " in ctx


def test_heartbeat_label_names_the_active_backend(monkeypatch):
    """Rehearsal: the heartbeat said "llm openai_compat" while answers said "ollama:qwen3.5:27b"."""
    from advisor.llm_engine import LLMEngine, OpenAICompatBackend

    remote = OpenAICompatBackend("http://127.0.0.1:9", model="qwen3.5:27b")
    monkeypatch.setattr(remote, "is_ollama", lambda: setattr(remote, "_is_ollama", True) or True)
    engine = LLMEngine(backend="openai_compat", remote=remote)
    assert engine.primary == "openai_compat"
    assert engine.active_label == "ollama:qwen3.5:27b"  # before any query, from the probe
    engine._down_until["openai_compat"] = time.monotonic() + 60
    assert engine.active_label in ("template:sitrep",) or engine.active_label.startswith("local:")
