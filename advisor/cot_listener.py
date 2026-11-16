"""Connect to TAK server TCP, parse CoT stream, maintain tactical state.

The connection is ``advisor.tak_client`` (kit contract v5). OpenTAKServer only relays CoT
to a TCP client that registered with a ``<contact>`` SA *after* OTS's internal RabbitMQ
channel for that connection opened, and never re-checks. So the client waits 3 s after each
connect, then registers as ``isrkit-tac-advisor-<ISR_HOST>`` / ``ADVISOR <ISR_HOST>``
(callsigns must be unique in OTS), and re-sends that SA every 60 s.
"""

import logging
import os
import random
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from advisor.tak_client import TakClient, build_sa

logger = logging.getLogger(__name__)

TAK_REPO, TAK_SHORT = "tac-advisor", "ADVISOR"
KIT_UID_PREFIX = "isrkit-"  # registration SAs of kit tools (contract v5)


@dataclass
class PositionInfo:
    callsign: str
    lat: float
    lon: float
    alt: float = 0.0
    speed: float | None = None
    heading: float | None = None
    battery: float | None = None
    timestamp: float = field(default_factory=time.time)
    label: str = ""  # "SIMULATED" | "RECORDED" | "" (live), contract C4.3


@dataclass
class MessageInfo:
    sender: str
    chatroom: str
    body: str
    timestamp: float = field(default_factory=time.time)
    label: str = ""  # "SIMULATED" | "RECORDED" | "" (live), contract C4.3


@dataclass
class AlertInfo:
    callsign: str
    alert_type: str
    lat: float
    lon: float
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    label: str = ""  # "SIMULATED" | "RECORDED" | "" (live), contract C4.3


@dataclass
class DroneInfo:
    uid: str
    lat: float
    lon: float
    alt: float = 0.0
    callsign: str = ""
    timestamp: float = field(default_factory=time.time)
    label: str = ""  # "SIMULATED" | "RECORDED" | "" (live), contract C4.3
    source: str = ""
    remarks: str = ""


@dataclass
class SensorEvent:
    """A detection reported by one of the sensor tools (acoustic, op-sentry, change-detect, rf-spectrum, ...)."""

    uid: str
    source: str
    cot_type: str
    callsign: str
    lat: float
    lon: float
    remarks: str = ""
    timestamp: float = field(default_factory=time.time)
    label: str = ""  # "SIMULATED" | "RECORDED" | "" (live), contract C4.3

    @property
    def affiliation(self) -> str:
        parts = self.cot_type.split("-")
        return {"h": "hostile", "u": "unknown", "n": "neutral", "s": "suspect", "j": "joker",
                "k": "faker"}.get(parts[1] if len(parts) > 1 else "", "unknown")


@dataclass
class SensorAsset:
    """A sensor's own position marker (e.g. an acoustic node or op-sentry OP)."""

    uid: str
    source: str
    callsign: str
    lat: float
    lon: float
    remarks: str = ""
    timestamp: float = field(default_factory=time.time)
    label: str = ""  # "SIMULATED" | "RECORDED" | "" (live), contract C4.3


# uid prefix -> tool name, for CoT from the ISR kit's sensor tools (case-insensitive)
SENSOR_UID_PREFIXES = {
    "acoustic-": "acoustic-detector",
    "opsentry-": "op-sentry",
    "op-sentry-": "op-sentry",
    "change-detect-": "change-detect",
    "changedetect-": "change-detect",
    "rfspec-": "rf-spectrum",
    "rf-spectrum-": "rf-spectrum",
    "fpv-vision-": "fpv-vision",
    "drone-": "drone-detector",
}


def data_label(uid: str = "", callsign: str = "", remarks: str = "") -> str:
    """SIMULATED / RECORDED / "" (live) for a CoT event, from the kit's labelling (contract C4.3):
    remarks starting "SIMULATED:" / "RECORDED:" (or "[SIM]" / "[REC]"), or a SIM- callsign/uid."""
    rem = remarks.lstrip().upper()
    if rem.startswith(("SIMULATED", "[SIM]")):
        return "SIMULATED"
    if rem.startswith(("RECORDED", "[REC]")):
        return "RECORDED"
    for v in (callsign.upper(), uid.upper()):
        if v.startswith("SIM-") or "-SIM-" in v:
            return "SIMULATED"
    return ""


def sensor_source(uid: str, remarks: str = "") -> str:
    """Which sensor tool produced this CoT, or "" if it does not look like one of ours."""
    low = uid.lower()
    for prefix, name in SENSOR_UID_PREFIXES.items():
        if low.startswith(prefix):
            return name
    rlow = remarks.lower()
    for name in ("acoustic-detector", "op-sentry", "change-detect", "rf-spectrum", "fpv-vision",
                 "drone-detector"):
        if name in rlow:
            return name
    return ""


def build_hello(uid: str, callsign: str, now: float | None = None) -> str:
    """The registration SA (contract v5), without a position. Kept for callers of the old name."""
    return build_sa(uid, callsign, TAK_REPO, None, None, now)


class CotListener:
    """Connects to TAK server TCP and maintains tactical state from CoT events."""

    POSITION_STALE_S = 60
    ALERT_STALE_S = 300
    EVENT_STALE_S = 900  # sensor detections stay in the picture for 15 min
    ASSET_STALE_S = 600
    MAX_MESSAGES = 50
    MAX_EVENTS = 100

    def __init__(self, host: str = "127.0.0.1", port: int = 8087, callsign: str | None = None):
        self._host = host
        self._port = port
        # callsigns are unique in OTS: the default "ADVISOR <station>" is; an override must be too
        self._client = TakClient(TAK_REPO, TAK_SHORT, host=host, port=port, on_event=self._process_event,
                                 callsign=callsign or os.environ.get("ADVISOR_CALLSIGN", "").strip() or None)
        self._callsign = self._client.callsign
        self._uid = self._client.uid
        self._lock = threading.Lock()
        self._positions: dict[str, PositionInfo] = {}
        self._messages: list[MessageInfo] = []
        self._alerts: list[AlertInfo] = []
        self._drones: dict[str, DroneInfo] = {}
        self._events: dict[str, SensorEvent] = {}
        self._assets: dict[str, SensorAsset] = {}
        self._connected = False
        self._running = False

    @property
    def connected(self) -> bool:
        """Registered with the TAK server (contract v5), not just TCP-connected."""
        return self._connected or self._client.registered.is_set()

    def start(self):
        self._running = True
        self._client.start()
        threading.Thread(target=self._cleanup_loop, daemon=True).start()

    def stop(self):
        self._running = False
        self._client.stop()

    def get_state(self) -> dict:
        with self._lock:
            return {
                "positions": dict(self._positions),
                "messages": list(self._messages),
                "alerts": list(self._alerts),
                "drones": list(self._drones.values()),
                "sensor_events": sorted(self._events.values(), key=lambda e: e.timestamp),
                "sensor_assets": list(self._assets.values()),
            }

    def _process_event(self, xml_str: str):
        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError:
            return

        cot_type = root.get("type", "")
        uid = root.get("uid", "")
        point = root.find("point")
        if point is None:
            return

        lat = float(point.get("lat", 0))
        lon = float(point.get("lon", 0))
        alt = float(point.get("hae", 0))
        detail = root.find("detail")

        if uid == self._uid:
            return  # our own registration SA

        if uid.startswith(KIT_UID_PREFIX):
            self._kit_registration(root, uid, lat, lon, point)
            return

        callsign = uid
        remarks_text = ""
        if detail is not None:
            contact = detail.find("contact")
            if contact is not None:
                callsign = contact.get("callsign", uid)
            remarks_el = detail.find("remarks")
            if remarks_el is not None and remarks_el.text:
                remarks_text = remarks_el.text.strip()
        source = sensor_source(uid, remarks_text)
        label = data_label(uid, callsign, remarks_text)
        parts = cot_type.split("-")
        is_atom = len(parts) >= 3 and parts[0] == "a"
        affiliation = parts[1] if is_atom else ""
        is_air = is_atom and parts[2] == "A"

        with self._lock:
            # UAVs from any source (drone-detector, acoustic, ...)
            if is_air and affiliation != "f":
                self._drones[uid] = DroneInfo(
                    uid=uid, lat=lat, lon=lon, alt=alt, callsign=callsign,
                    source=source, remarks=remarks_text, label=label,
                )

            # Sensor tools' own position markers (acoustic node, OP, ...): assets, not troops
            elif source and is_atom and affiliation == "f":
                self._assets[uid] = SensorAsset(
                    uid=uid, source=source, callsign=callsign, lat=lat, lon=lon, remarks=remarks_text,
                    label=label,
                )

            # Detections: hostile/unknown/suspect ground or sea contacts, or any atom from our sensors
            elif is_atom and (affiliation in ("h", "u", "s", "j", "k", "n") or source):
                self._events[uid] = SensorEvent(
                    uid=uid, source=source or "TAK", cot_type=cot_type, callsign=callsign,
                    lat=lat, lon=lon, remarks=remarks_text, label=label,
                )
                if len(self._events) > self.MAX_EVENTS:
                    oldest = min(self._events.values(), key=lambda e: e.timestamp)
                    del self._events[oldest.uid]

            # PLI — friendly position
            elif cot_type.startswith("a-f-G-"):
                speed = None
                heading = None
                battery = None
                if detail is not None:
                    track = detail.find("track")
                    if track is not None:
                        speed = float(track.get("speed", 0))
                        heading = float(track.get("course", 0))
                    status = detail.find("status")
                    if status is not None and status.get("battery"):
                        battery = float(status.get("battery"))

                self._positions[callsign] = PositionInfo(
                    callsign=callsign, lat=lat, lon=lon, alt=alt,
                    speed=speed, heading=heading, battery=battery, label=label,
                )

            # GeoChat
            elif cot_type == "b-t-f":
                body = ""
                chatroom = "All Chat Rooms"
                if detail is not None:
                    remarks = detail.find("remarks")
                    if remarks is not None:
                        body = remarks.text or ""
                    chat = detail.find("__chat")
                    if chat is not None:
                        chatroom = chat.get("chatroom", chatroom)
                        callsign = chat.get("senderCallsign", callsign)

                self._messages.append(MessageInfo(
                    sender=callsign, chatroom=chatroom, body=body, label=data_label(uid, callsign, body),
                ))
                if len(self._messages) > self.MAX_MESSAGES:
                    self._messages = self._messages[-self.MAX_MESSAGES:]

            # Emergency
            elif cot_type.startswith("b-a-o-"):
                message = ""
                if detail is not None:
                    remarks = detail.find("remarks")
                    if remarks is not None:
                        message = remarks.text or ""
                self._alerts.append(AlertInfo(
                    callsign=callsign, alert_type=cot_type,
                    lat=lat, lon=lon, message=message, label=data_label(uid, callsign, message),
                ))

            # Spot reports / map points from a sensor tool (e.g. fpv-vision SPOTREP marker)
            elif source and cot_type.startswith("b-m-p"):
                self._events[uid] = SensorEvent(
                    uid=uid, source=source, cot_type=cot_type, callsign=callsign,
                    lat=lat, lon=lon, remarks=remarks_text, label=label,
                )

    def _kit_registration(self, root, uid: str, lat: float, lon: float, point):
        """A kit tool's registration SA: a sensor asset if it carries a position, else ignored."""
        if point.get("ce") == "9999999" or (lat == 0 and lon == 0):
            return  # no position set (contract C5.2): never drawn, not a unit either
        takv = root.find("detail/takv")
        contact = root.find("detail/contact")
        source = (takv.get("device") if takv is not None else "") or uid[len(KIT_UID_PREFIX):]
        callsign = contact.get("callsign", uid) if contact is not None else uid
        with self._lock:
            self._assets[uid] = SensorAsset(uid=uid, source=source, callsign=callsign, lat=lat, lon=lon,
                                            remarks="kit tool station")

    def _cleanup_loop(self):
        while self._running:
            time.sleep(15)
            now = time.time()
            with self._lock:
                stale_keys = [
                    k for k, v in self._positions.items()
                    if now - v.timestamp > self.POSITION_STALE_S
                ]
                for k in stale_keys:
                    del self._positions[k]

                self._alerts = [
                    a for a in self._alerts
                    if now - a.timestamp <= self.ALERT_STALE_S
                ]

                stale_drones = [
                    k for k, v in self._drones.items()
                    if now - v.timestamp > self.ALERT_STALE_S
                ]
                for k in stale_drones:
                    del self._drones[k]

                for store, max_age in ((self._events, self.EVENT_STALE_S), (self._assets, self.ASSET_STALE_S)):
                    for k in [k for k, v in store.items() if now - v.timestamp > max_age]:
                        del store[k]


class SimulatedCotListener(CotListener):
    """Generates fake tactical data for testing without a TAK server."""

    UNITS = [
        ("Alpha-1", 51.15, -1.75),
        ("Bravo-2", 51.152, -1.748),
        ("Charlie-3", 51.148, -1.752),
        ("Delta-4", 51.151, -1.746),
        ("Echo-5", 51.149, -1.754),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._connected = True

    def start(self):
        self._running = True
        threading.Thread(target=self._simulate_loop, daemon=True).start()
        threading.Thread(target=self._cleanup_loop, daemon=True).start()
        logger.info("Simulation mode — generating fake tactical data")

    def _simulate_loop(self):
        offsets = {name: (0.0, 0.0) for name, _, _ in self.UNITS}

        while self._running:
            # Update positions with drift
            with self._lock:
                for name, base_lat, base_lon in self.UNITS:
                    dx, dy = offsets[name]
                    dx += random.uniform(-0.0002, 0.0002)
                    dy += random.uniform(-0.0002, 0.0002)
                    dx = max(-0.005, min(0.005, dx))
                    dy = max(-0.005, min(0.005, dy))
                    offsets[name] = (dx, dy)

                    self._positions[name] = PositionInfo(
                        callsign=name,
                        lat=base_lat + dx,
                        lon=base_lon + dy,
                        speed=random.uniform(0, 8),
                        heading=random.uniform(0, 360),
                        battery=random.uniform(40, 100),
                        label="SIMULATED",
                    )

            # Occasional messages
            if random.random() < 0.15:
                sender = random.choice(self.UNITS)[0]
                messages = [
                    "Contact, 200m north",
                    "Moving to checkpoint bravo",
                    "All clear, holding position",
                    "Request resupply",
                    "Eyes on two pax, grid 51.150 -1.749",
                    "Roger, moving now",
                    "Casualty — need CASEVAC",
                    "Comms check",
                ]
                with self._lock:
                    self._messages.append(MessageInfo(
                        sender=sender, chatroom="All Chat Rooms",
                        body=random.choice(messages), label="SIMULATED",
                    ))
                    if len(self._messages) > self.MAX_MESSAGES:
                        self._messages = self._messages[-self.MAX_MESSAGES:]

            # Rare alerts
            if random.random() < 0.02:
                unit = random.choice(self.UNITS)
                with self._lock:
                    self._alerts.append(AlertInfo(
                        callsign=unit[0], alert_type="b-a-o-tbl",
                        lat=unit[1], lon=unit[2], message="SOS — contact", label="SIMULATED",
                    ))

            # Sensor tool events (as they would arrive from OTS)
            if random.random() < 0.05:
                kind = random.choice(["acoustic", "opsentry", "change", "rf"])
                lat = 51.15 + random.uniform(-0.004, 0.004)
                lon = -1.75 + random.uniform(-0.004, 0.004)
                ev = {
                    "acoustic": ("ACOUSTIC-north-UAV", "acoustic-detector", "a-u-A", "UAV? (acoustic north)",
                                 "Multirotor acoustic signature, confidence 0.82. Prop tone ~180 Hz."),
                    "opsentry": ("opsentry-OP1-7", "op-sentry", "a-u-G-U-C-I", "OP1 2xPERSON BRG 045",
                                 "op-sentry: 2 person(s) entering zone TRACK, bearing 045."),
                    "change": ("change-detect-a1b2", "change-detect", "a-u-G", "CHANGE a1b2 (3)",
                               "Change detected: new vehicle tracks at the gate [change-detect a1b2]"),
                    "rf": ("RFSPEC-433-920", "rf-spectrum", "a-u-G", "RF 433.920",
                           "New emitter 433.920 MHz, -48 dBm, bursty (likely ISM telemetry)."),
                }[kind]
                with self._lock:
                    if ev[2].startswith("a-u-A"):
                        self._drones[ev[0]] = DroneInfo(uid=ev[0], lat=lat, lon=lon, alt=40.0,
                                                        callsign=ev[3], source=ev[1], remarks=ev[4],
                                                        label="SIMULATED")
                    else:
                        self._events[ev[0]] = SensorEvent(uid=ev[0], source=ev[1], cot_type=ev[2],
                                                          callsign=ev[3], lat=lat, lon=lon, remarks=ev[4],
                                                          label="SIMULATED")

            # Rare drone sighting
            if random.random() < 0.03:
                with self._lock:
                    self._drones["DRONE-sim-1"] = DroneInfo(
                        uid="DRONE-sim-1",
                        lat=51.151 + random.uniform(-0.002, 0.002),
                        lon=-1.749 + random.uniform(-0.002, 0.002),
                        alt=random.uniform(30, 120),
                        callsign="Unknown UAV",
                        source="drone-detector",
                        remarks="DRONE_CONFIRMED via DJI DroneID, serial 1581F4XXXX (simulated)",
                        label="SIMULATED",
                    )

            time.sleep(2)