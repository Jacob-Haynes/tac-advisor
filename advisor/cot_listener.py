"""Connect to FreeTAKServer TCP, parse CoT stream, maintain tactical state."""

import logging
import random
import socket
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CotStreamParser:
    """Stateful parser that extracts complete CoT <event> elements from a TCP byte stream."""

    def __init__(self):
        self._buf = ""

    def feed(self, data: str) -> list[str]:
        self._buf += data
        events = []
        while "</event>" in self._buf:
            end = self._buf.index("</event>") + len("</event>")
            chunk = self._buf[:end].strip()
            self._buf = self._buf[end:]
            start = chunk.rfind("<event")
            if start >= 0:
                events.append(chunk[start:])
        return events


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


@dataclass
class MessageInfo:
    sender: str
    chatroom: str
    body: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AlertInfo:
    callsign: str
    alert_type: str
    lat: float
    lon: float
    message: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class DroneInfo:
    uid: str
    lat: float
    lon: float
    alt: float = 0.0
    callsign: str = ""
    timestamp: float = field(default_factory=time.time)


class CotListener:
    """Connects to FTS TCP and maintains tactical state from CoT events."""

    POSITION_STALE_S = 60
    ALERT_STALE_S = 300
    MAX_MESSAGES = 50
    RECONNECT_MIN = 2
    RECONNECT_MAX = 60

    def __init__(self, host: str = "127.0.0.1", port: int = 8087):
        self._host = host
        self._port = port
        self._lock = threading.Lock()
        self._positions: dict[str, PositionInfo] = {}
        self._messages: list[MessageInfo] = []
        self._alerts: list[AlertInfo] = []
        self._drones: dict[str, DroneInfo] = {}
        self._connected = False
        self._running = False

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self):
        self._running = True
        threading.Thread(target=self._reader_loop, daemon=True).start()
        threading.Thread(target=self._cleanup_loop, daemon=True).start()

    def stop(self):
        self._running = False

    def get_state(self) -> dict:
        with self._lock:
            return {
                "positions": dict(self._positions),
                "messages": list(self._messages),
                "alerts": list(self._alerts),
                "drones": list(self._drones.values()),
            }

    def _reader_loop(self):
        backoff = self.RECONNECT_MIN
        parser = CotStreamParser()

        while self._running:
            sock = None
            try:
                sock = socket.create_connection((self._host, self._port), timeout=10)
                self._connected = True
                backoff = self.RECONNECT_MIN
                logger.info("Connected to FTS at %s:%d", self._host, self._port)

                while self._running:
                    data = sock.recv(4096)
                    if not data:
                        break
                    for event_xml in parser.feed(data.decode("utf-8", errors="replace")):
                        self._process_event(event_xml)

            except (OSError, ConnectionError) as e:
                logger.warning("FTS connection error: %s (retry in %ds)", e, backoff)
            finally:
                self._connected = False
                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass

            if self._running:
                time.sleep(backoff)
                backoff = min(backoff * 2, self.RECONNECT_MAX)

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

        callsign = uid
        if detail is not None:
            contact = detail.find("contact")
            if contact is not None:
                callsign = contact.get("callsign", uid)

        with self._lock:
            # PLI — friendly position
            if cot_type.startswith("a-f-G-"):
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
                    speed=speed, heading=heading, battery=battery,
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
                    sender=callsign, chatroom=chatroom, body=body,
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
                    lat=lat, lon=lon, message=message,
                ))

            # Hostile UAV (drone detector)
            elif cot_type.startswith("a-h-A-"):
                self._drones[uid] = DroneInfo(
                    uid=uid, lat=lat, lon=lon, alt=alt, callsign=callsign,
                )

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


class SimulatedCotListener(CotListener):
    """Generates fake tactical data for testing without FTS."""

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
                        body=random.choice(messages),
                    ))
                    if len(self._messages) > self.MAX_MESSAGES:
                        self._messages = self._messages[-self.MAX_MESSAGES:]

            # Rare alerts
            if random.random() < 0.02:
                unit = random.choice(self.UNITS)
                with self._lock:
                    self._alerts.append(AlertInfo(
                        callsign=unit[0], alert_type="b-a-o-tbl",
                        lat=unit[1], lon=unit[2], message="SOS — contact",
                    ))

            # Rare drone sighting
            if random.random() < 0.03:
                with self._lock:
                    self._drones["DRONE-sim-1"] = DroneInfo(
                        uid="DRONE-sim-1",
                        lat=51.151 + random.uniform(-0.002, 0.002),
                        lon=-1.749 + random.uniform(-0.002, 0.002),
                        alt=random.uniform(30, 120),
                        callsign="Unknown UAV",
                    )

            time.sleep(2)