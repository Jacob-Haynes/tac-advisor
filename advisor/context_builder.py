"""Build tactical text context from CoT listener state for LLM prompting."""

import time

SYSTEM_PROMPT = (
    "You are a tactical advisor for a UK Army Blue Force Tracker system. You provide concise, "
    "actionable suggestions using UK military terminology and doctrine. You have access to the "
    "current operational picture including unit positions, alerts, messages, drone detections and "
    "reports from ISR sensors (acoustic, camera OP, change detection, RF spectrum). Treat sensor "
    "detections as unconfirmed unless corroborated, and say which sensor reported them. "
    "Keep responses brief and formatted for quick reading. Use MGRS grid references where provided. "
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


def build_context(state: dict) -> str:
    """Build a text context block from CotListener state snapshot."""
    sections: list[str] = []

    # Unit positions
    positions = state.get("positions", {})
    if positions:
        lines = ["UNIT POSITIONS:"]
        for p in positions.values():
            parts = [f"  {p.callsign}{_tag(p)}: lat={p.lat:.5f} lon={p.lon:.5f}"]
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
                f"  {a.callsign}{_tag(a)} at lat={a.lat:.5f} lon={a.lon:.5f}{msg} "
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
                f"  {d.callsign or d.uid}{src}{_tag(d)}: lat={d.lat:.5f} lon={d.lon:.5f} "
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
                f"  [{e.source}] {e.callsign}{_tag(e)} ({e.affiliation}): lat={e.lat:.5f} lon={e.lon:.5f} "
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
                f"  [{a.source}] {a.callsign}{_tag(a)}: lat={a.lat:.5f} lon={a.lon:.5f} "
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

    if block:
        sections.insert(0, block)
    return "\n\n".join(sections)