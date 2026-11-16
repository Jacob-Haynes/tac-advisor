import socket
import threading
import time

from advisor.context_builder import build_context
from advisor.cot_listener import CotListener, build_hello, sensor_source


def ev(uid, cot_type, callsign, remarks="", lat=51.15, lon=-1.75, extra=""):
    return (
        f'<event version="2.0" uid="{uid}" type="{cot_type}" how="m-g" time="2026-11-16T12:00:00Z" '
        f'start="2026-11-16T12:00:00Z" stale="2026-11-16T12:05:00Z">'
        f'<point lat="{lat}" lon="{lon}" hae="10" ce="10" le="10"/>'
        f'<detail><contact callsign="{callsign}"/><remarks>{remarks}</remarks>{extra}</detail></event>'
    )


# Formats as emitted by the sibling sensor tools
ACOUSTIC_UAV = ev("ACOUSTIC-north-UAV", "a-u-A-M-H-Q", "UAV? (acoustic north)",
                  "Multirotor signature conf 0.82. Prop tone ~180 Hz.")
ACOUSTIC_NODE = ev("ACOUSTIC-north", "a-f-G-E-S", "ACOUSTIC north", "acoustic-detector node")
OPSENTRY_PERSON = ev("opsentry-OP1-12", "a-u-G-U-C-I", "OP1 2xPERSON BRG 045", "2 person(s) entering zone TRACK")
OPSENTRY_OP = ev("opsentry-OP1", "a-f-G-E-S", "OP1", "op-sentry camera OP.")
CHANGE = ev("change-detect-a1b2c3", "a-u-G", "CHANGE b2c3 (3)", "Change detected: vehicle tracks at gate")
RF = ev("RFSPEC-433-920", "a-u-G", "RF 433.920", "New emitter -48 dBm")
DRONE = ev("DRONE-60601FAABBCC", "a-h-A-M-F-Q", "1581F4ABC", "DRONE_CONFIRMED")
PLI = ev("ANDROID-abc", "a-f-G-U-C", "Alpha-1", extra='<track speed="1.0" course="90"/>')
CHAT = ('<event version="2.0" uid="GeoChat.x" type="b-t-f" how="h-g-i-g-o" time="t" start="t" stale="t">'
        '<point lat="0" lon="0" hae="0" ce="0" le="0"/><detail><__chat chatroom="All Chat Rooms" '
        'senderCallsign="Bravo-2"/><remarks>Contact north</remarks></detail></event>')


def test_sensor_source_by_prefix_and_remarks():
    assert sensor_source("ACOUSTIC-x") == "acoustic-detector"
    assert sensor_source("opsentry-OP1-1") == "op-sentry"
    assert sensor_source("change-detect-1") == "change-detect"
    assert sensor_source("RFSPEC-1") == "rf-spectrum"
    assert sensor_source("x", "seen by op-sentry") == "op-sentry"
    assert sensor_source("ANDROID-1") == ""


def test_classifies_sibling_tool_cot():
    li = CotListener()
    for x in (ACOUSTIC_UAV, ACOUSTIC_NODE, OPSENTRY_PERSON, OPSENTRY_OP, CHANGE, RF, DRONE, PLI, CHAT):
        li._process_event(x)
    st = li.get_state()
    assert set(st["positions"]) == {"Alpha-1"}  # sensors are not counted as troops
    assert {d.uid: d.source for d in st["drones"]} == {
        "ACOUSTIC-north-UAV": "acoustic-detector", "DRONE-60601FAABBCC": "drone-detector"}
    assert {e.source for e in st["sensor_events"]} == {"op-sentry", "change-detect", "rf-spectrum"}
    assert {a.uid for a in st["sensor_assets"]} == {"ACOUSTIC-north", "opsentry-OP1"}
    assert st["messages"][0].sender == "Bravo-2"


def test_own_hello_is_ignored():
    li = CotListener()
    li._process_event(build_hello(li._uid, "TAC-ADVISOR"))
    assert li.get_state()["positions"] == {}


def test_sitrep_context_mentions_sensors():
    li = CotListener()
    for x in (ACOUSTIC_UAV, OPSENTRY_PERSON, CHANGE, RF, ACOUSTIC_NODE):
        li._process_event(x)
    ctx = build_context(li.get_state())
    assert "SENSOR DETECTIONS" in ctx and "SENSORS ONLINE" in ctx
    for name in ("op-sentry", "change-detect", "rf-spectrum", "acoustic-detector"):
        assert name in ctx
    assert "OP1 2xPERSON BRG 045" in ctx and "Prop tone" in ctx


class FakeOTS:
    """Mimics OpenTAKServer's TCP port: streams CoT to a client only after it sends a contact event."""

    def __init__(self, to_send):
        self.to_send = to_send
        self.connections = 0
        self.received = b""
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(5)
        self.port = self.srv.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def _client(self, conn):
        buf = b""
        while b"<contact" not in buf or b"</event>" not in buf:
            data = conn.recv(4096)
            if not data:
                return
            buf += data
        self.received += buf
        for x in self.to_send:
            conn.sendall(x.encode())
        try:
            while conn.recv(4096):
                pass
        except OSError:
            pass


def _wait(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_listener_receives_from_ots_after_hello_and_survives_quiet_periods():
    ots = FakeOTS([PLI, OPSENTRY_PERSON])
    li = CotListener(port=ots.port)
    li.RECV_TIMEOUT_S = 0.2
    li.start()
    try:
        assert _wait(lambda: "Alpha-1" in li.get_state()["positions"])
        assert b'platform="tac-advisor"' in ots.received
        time.sleep(1.0)  # several recv timeouts with no traffic
        assert ots.connections == 1 and li.connected
    finally:
        li.stop()
        ots.srv.close()
