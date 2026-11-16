"""Offline SITREP built straight from the CoT picture, with no model at all.

The last resort in every backend order: when Ollama, Gemini and the local GGUF model are all
unavailable, a query still gets a factual summary instead of a setup or error message.
It counts unit positions, alerts by type, drone and sensor detections by source, and when each
was last seen, and it carries the SIMULATED / RECORDED labels through (contract C4.3), e.g.
"2 simulated drone detections".
"""

from __future__ import annotations

import time
from collections import Counter

SIMULATED, RECORDED = "SIMULATED", "RECORDED"
ACTIVE_S = 60
MESH_LIMIT = 200


def _ago(ts: float | None, now: float) -> str:
    if not ts:
        return "never"
    diff = max(0, int(now - ts))
    if diff < 60:
        return f"{diff}s ago"
    if diff < 3600:
        return f"{diff // 60}min ago"
    return f"{diff // 3600}h ago"


def _plural(n: int, noun: str) -> str:
    return noun if n == 1 else (noun[:-1] + "ies" if noun.endswith("y") else noun + "s")


def count_phrase(items: list, noun: str) -> str:
    """'2 simulated drone detections', '3 drone detections (2 simulated, 1 recorded)', '1 unit'."""
    n = len(items)
    labels = Counter(getattr(i, "label", "") or "" for i in items)
    words = _plural(n, noun)
    if n and labels.get(SIMULATED) == n:
        return f"{n} simulated {words}"
    if n and labels.get(RECORDED) == n:
        return f"{n} recorded {words}"
    extra = [f"{labels[k]} {k.lower()}" for k in (SIMULATED, RECORDED) if labels.get(k)]
    return f"{n} {words}" + (f" ({', '.join(extra)})" if extra else "")


def _by_source(items: list) -> str:
    counts = Counter(getattr(i, "source", "") or "TAK" for i in items)
    return ", ".join(f"{src} {n}" for src, n in counts.most_common())


def _alert_name(alert_type: str) -> str:
    return {"b-a-o-tbl": "SOS", "b-a-o-pan": "ring the bell", "b-a-o-opn": "troops in contact",
            "b-a-o-can": "cancelled"}.get(alert_type, alert_type or "alert")


def _newest(*groups) -> float | None:
    stamps = [getattr(i, "timestamp", 0) for g in groups for i in g]
    return max(stamps) if stamps else None


def build_sitrep(state: dict, compact: bool = False, now: float | None = None) -> str:
    now = time.time() if now is None else now
    positions = list((state.get("positions") or {}).values())
    alerts = list(state.get("alerts") or [])
    drones = list(state.get("drones") or [])
    events = list(state.get("sensor_events") or [])
    assets = list(state.get("sensor_assets") or [])
    active = [p for p in positions if now - p.timestamp <= ACTIVE_S]
    last = _newest(positions, alerts, drones, events, assets)
    stamp = time.strftime("%H%MZ", time.gmtime(now))

    if compact:
        parts = [f"SITREP {stamp} (offline, no LLM)"]
        parts.append(count_phrase(positions, "unit") + (f", {len(active)} active" if positions else ""))
        if alerts:
            kinds = Counter(_alert_name(a.alert_type) for a in alerts)
            parts.append(count_phrase(alerts, "alert") + ": " + ", ".join(f"{k} {n}" for k, n in kinds.items()))
        if drones:
            parts.append(count_phrase(drones, "drone detection") + f" ({_by_source(drones)})")
        if events:
            parts.append(count_phrase(events, "sensor detection") + f" ({_by_source(events)})")
        if not (positions or alerts or drones or events):
            parts.append("no picture received")
        parts.append(f"last {_ago(last, now)}")
        text = "; ".join(parts)
        return text if len(text) <= MESH_LIMIT else text[: MESH_LIMIT - 3] + "..."

    lines = [f"OFFLINE SITREP {stamp} (no LLM available: summary built from the CoT picture)"]
    lines.append(f"Units: {count_phrase(positions, 'unit')}, {len(active)} active in the last {ACTIVE_S}s, "
                 f"last report {_ago(_newest(positions), now)}")
    if alerts:
        kinds = Counter(_alert_name(a.alert_type) for a in alerts)
        lines.append(f"Alerts: {count_phrase(alerts, 'alert')}: "
                     + ", ".join(f"{k} {n}" for k, n in kinds.items())
                     + f"; latest {_ago(_newest(alerts), now)} ({alerts[-1].callsign})")
    else:
        lines.append("Alerts: none")
    if drones:
        lines.append(f"Drones: {count_phrase(drones, 'drone detection')} by source: {_by_source(drones)}; "
                     f"latest {_ago(_newest(drones), now)}")
    else:
        lines.append("Drones: none")
    if events:
        lines.append(f"Sensors: {count_phrase(events, 'sensor detection')} by source: {_by_source(events)}; "
                     f"latest {_ago(_newest(events), now)}")
    if assets:
        lines.append(f"Sensors online: {len(assets)} ({_by_source(assets)})")
    if any(getattr(i, "label", "") for i in positions + alerts + drones + events):
        lines.append("Note: items counted as simulated or recorded are exercise data, not live.")
    lines.append(f"Picture last updated {_ago(last, now)}.")
    return "\n".join(lines)


class TemplateBackend:
    """LLMEngine backend that ignores the prompt and summarises the picture (never fails)."""

    name = "template"
    model_name = "sitrep"
    available = True

    def __init__(self, state_fn=None):
        self.state_fn = state_fn

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        state = {}
        if self.state_fn is not None:
            try:
                state = self.state_fn() or {}
            except Exception:
                state = {}
        return build_sitrep(state, compact=max_tokens <= 200)
