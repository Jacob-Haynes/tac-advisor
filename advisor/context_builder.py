"""Build tactical text context from CoT listener state for LLM prompting."""

import time

SYSTEM_PROMPT = (
    "You are a tactical advisor for a UK Army Blue Force Tracker system. You provide concise, "
    "actionable suggestions using UK military terminology and doctrine. You have access to the "
    "current operational picture including unit positions, alerts, messages, and drone detections. "
    "Keep responses brief and formatted for quick reading. Use MGRS grid references where provided. "
    "Refer to units by callsign. Prioritise safety and force protection."
)


def _relative_time(ts: float) -> str:
    diff = int(time.time() - ts)
    if diff < 60:
        return f"{diff}s ago"
    if diff < 3600:
        return f"{diff // 60}min ago"
    return f"{diff // 3600}h ago"


def build_context(state: dict) -> str:
    """Build a text context block from CotListener state snapshot."""
    sections: list[str] = []

    # Unit positions
    positions = state.get("positions", {})
    if positions:
        lines = ["UNIT POSITIONS:"]
        for p in positions.values():
            parts = [f"  {p.callsign}: lat={p.lat:.5f} lon={p.lon:.5f}"]
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
                f"  {a.callsign} at lat={a.lat:.5f} lon={a.lon:.5f}{msg} "
                f"({_relative_time(a.timestamp)})"
            )
        sections.append("\n".join(lines))

    # Drone detections
    drones = state.get("drones", [])
    if drones:
        lines = ["DRONE DETECTIONS:"]
        for d in drones:
            lines.append(
                f"  {d.callsign or d.uid}: lat={d.lat:.5f} lon={d.lon:.5f} "
                f"alt={d.alt:.0f}m ({_relative_time(d.timestamp)})"
            )
        sections.append("\n".join(lines))

    # Recent messages
    messages = state.get("messages", [])
    recent = messages[-20:] if len(messages) > 20 else messages
    if recent:
        lines = ["RECENT MESSAGES:"]
        for m in recent:
            lines.append(
                f"  {m.sender} -> {m.chatroom}: {m.body} "
                f"({_relative_time(m.timestamp)})"
            )
        sections.append("\n".join(lines))

    if not sections:
        return "No tactical data available."

    return "\n\n".join(sections)