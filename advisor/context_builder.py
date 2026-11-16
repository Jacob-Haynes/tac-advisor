"""Build tactical text context from CoT listener state for LLM prompting."""

import os
import time

from advisor.grid import bearing_distance, grid_format, grid_name, grid_ref, relative

SYSTEM_PROMPT = (
    "You are a tactical advisor for a UK Army Blue Force Tracker system. You provide concise, "
    "actionable suggestions using UK military terminology and doctrine. You have access to the "
    "current operational picture including unit positions, alerts, messages, drone detections and "
    "reports from ISR sensors (acoustic, camera OP, change detection, RF spectrum). Treat sensor "
    "detections as unconfirmed unless corroborated, and say which sensor reported them. "
    "Keep responses brief and formatted for quick reading. "
    "Positions are given as grid references in the format named on the GRID line. Copy each grid "
    "reference exactly as written, character for character; never compute, shorten, round or invent "
    "one, and never turn other numbers into a grid reference. If an item says 'no position', say so. "
    "Relative positions (e.g. '1.2 km NW of HANGOVER') may be quoted as given. "
    "Refer to units by callsign. Prioritise safety and force protection. "
    "Items tagged [SIMULATED] or [RECORDED] are exercise or replayed data, not live: always say so "
    "when you report them (e.g. '2 simulated drone detections') and never present them as real. "
    "PICTURE COUNTS lists every category in the picture: a SITREP must mention every one of them "
    "with its count, even briefly; never leave a category out."
)


MAX_EVENTS_IN_CONTEXT = 15
REMARKS_MAX = 140


def _clip(text: str, limit: int = REMARKS_MAX) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _relative_time(ts: float) -> str:
    diff = int(time.time() - ts)
    if diff < 60:
        return f"{diff}s ago"
    if diff < 3600:
        return f"{diff // 60}min ago"
    return f"{diff // 3600}h ago"


def _tag(item) -> str:
    label = getattr(item, "label", "")
    return f" [{label}]" if label else ""


def _has_pos(item) -> bool:
    lat, lon = getattr(item, "lat", None), getattr(item, "lon", None)
    return lat is not None and lon is not None and not (lat == 0 and lon == 0)


def _op_position() -> tuple[float, float] | None:
    """The shared OP position (SENSOR_LAT/SENSOR_LON), if configured."""
    try:
        lat = float(os.environ.get("SENSOR_LAT", "").strip())
        lon = float(os.environ.get("SENSOR_LON", "").strip())
    except ValueError:
        return None
    return (lat, lon) if (lat or lon) else None


class _Locator:
    """Grid ref plus "1.2 km NW of <nearest friendly unit>" (or of the OP) for each item."""

    def __init__(self, positions: list):
        self.fmt = grid_format()
        self.units = [p for p in positions if _has_pos(p)]
        self.op = _op_position()

    def __call__(self, item, own_callsign: str | None = None) -> str:
        if not _has_pos(item):
            return "grid=no position"
        text = f"grid={grid_ref(item.lat, item.lon, self.fmt)}"
        refs = [(bearing_distance(u.lat, u.lon, item.lat, item.lon)[0], u.lat, u.lon, u.callsign)
                for u in self.units if u.callsign != own_callsign]
        if refs:
            _, rlat, rlon, name = min(refs)
            return f"{text} ({relative(item.lat, item.lon, rlat, rlon, name)})"
        if self.op:
            return f"{text} ({relative(item.lat, item.lon, *self.op, 'the OP')})"
        return text


def build_context(state: dict) -> str:
    """Build a text context block from CotListener state snapshot."""
    sections: list[str] = []

    positions = state.get("positions", {})
    where = _Locator(list(positions.values()))

    # Unit positions
    if positions:
        lines = ["UNIT POSITIONS:"]
        for p in positions.values():
            parts = [f"  {p.callsign}{_tag(p)}: {where(p, own_callsign=p.callsign)}"]
            if p.speed is not None:
                parts.append(f"speed={p.speed:.1f}m/s")
            if p.heading is not None:
                parts.append(f"hdg={p.heading:.0f}")
            if p.battery is not None:
                parts.append(f"batt={p.battery:.0f}%")
            status = "STALE" if time.time() - p.timestamp > 60 else "ACTIVE"
            parts.append(status)
            parts.append(_relative_time(p.timestamp))
            lines.append(", ".join(parts))
        sections.append("\n".join(lines))

    # Active alerts
    alerts = state.get("alerts", [])
    if alerts:
        lines = ["ACTIVE ALERTS:"]
        for a in alerts:
            msg = f": {a.message}" if a.message else ""
            lines.append(
                f"  {a.callsign}{_tag(a)} at {where(a, own_callsign=a.callsign)}{msg} "
                f"({_relative_time(a.timestamp)})"
            )
        sections.append("\n".join(lines))

    # Drone detections
    drones = state.get("drones", [])
    if drones:
        lines = ["DRONE DETECTIONS:"]
        for d in drones:
            src = f" [{d.source}]" if getattr(d, "source", "") else ""
            rem = f" — {_clip(d.remarks)}" if getattr(d, "remarks", "") else ""
            lines.append(
                f"  {d.callsign or d.uid}{src}{_tag(d)}: {where(d)} "
                f"alt={d.alt:.0f}m ({_relative_time(d.timestamp)}){rem}"
            )
        sections.append("\n".join(lines))

    # Sensor detections (acoustic, op-sentry, change-detect, rf-spectrum, ... via OTS)
    events = state.get("sensor_events", [])
    recent_events = events[-MAX_EVENTS_IN_CONTEXT:]
    if recent_events:
        lines = ["SENSOR DETECTIONS (unconfirmed, newest last):"]
        for e in recent_events:
            rem = f" — {_clip(e.remarks)}" if e.remarks else ""
            lines.append(
                f"  [{e.source}] {e.callsign}{_tag(e)} ({e.affiliation}): {where(e)} "
                f"({_relative_time(e.timestamp)}){rem}"
            )
        if len(events) > len(recent_events):
            lines.append(f"  (+{len(events) - len(recent_events)} older detections not shown)")
        sections.append("\n".join(lines))

    # Sensor assets (where our sensors are)
    assets = state.get("sensor_assets", [])
    if assets:
        lines = ["SENSORS ONLINE:"]
        for a in assets:
            lines.append(
                f"  [{a.source}] {a.callsign}{_tag(a)}: {where(a)} "
                f"(last report {_relative_time(a.timestamp)})"
            )
        sections.append("\n".join(lines))

    # Recent messages
    messages = state.get("messages", [])
    recent = messages[-20:] if len(messages) > 20 else messages
    if recent:
        lines = ["RECENT MESSAGES:"]
        for m in recent:
            lines.append(
                f"  {m.sender}{_tag(m)} -> {m.chatroom}: {m.body} "
                f"({_relative_time(m.timestamp)})"
            )
        sections.append("\n".join(lines))

    from advisor.template_sitrep import counts_block

    block = counts_block(state)
    if not sections:
        return "No tactical data available."

    from advisor.template_sitrep import count_phrase

    flagged = [i for group in (list(positions.values()), alerts, drones, events, messages) for i in group
               if getattr(i, "label", "")]
    if flagged:
        summary = []
        if drones:
            summary.append(count_phrase(drones, "drone detection"))
        if events:
            summary.append(count_phrase(events, "sensor detection"))
        if positions:
            summary.append(count_phrase(list(positions.values()), "unit position"))
        sections.insert(0, "DATA LABELS: some of this picture is SIMULATED or RECORDED (exercise data). "
                           "Counts: " + "; ".join(summary) + ".")

    sections.insert(0, f"GRID: {grid_name(where.fmt)}. Copy grid references verbatim; never invent one.")
    if block:
        sections.insert(0, block)
    return "\n\n".join(sections)