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


def _wait(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_listener_registers_late_enough_to_be_bound_and_receives(monkeypatch):
    """Against a fake OTS with the real rules (FK drop, bind delay, unique callsign), contract v5."""
    from advisor.tak_client import TakClient
    from tests.fake_ots import FakeOTS

    monkeypatch.setenv("TAK_REGISTER_DELAY_S", "0.5")
    monkeypatch.setenv("ISR_HOST", "isr-pi")
    ots = FakeOTS(bind_delay=0.3)
    li = CotListener(port=ots.port)
    li.start()
    sender = TakClient("op-sentry", "OPSENTRY", port=ots.port, station="isr-pi", sensor_lat=51.15,
                       sensor_lon=-1.75)
    try:
        assert _wait(lambda: li.connected)
        assert ots.euds.get("isrkit-tac-advisor-isr-pi") == "ADVISOR isr-pi"
        sender.start()
        sender.send(PLI)
        sender.send(OPSENTRY_PERSON)
        assert _wait(lambda: "Alpha-1" in li.get_state()["positions"])
        assert _wait(lambda: any(e.source == "op-sentry" for e in li.get_state()["sensor_events"]))
        # the sender's own registration SA shows up as a sensor station, not as a friendly unit
        assets = li.get_state()["sensor_assets"]
        assert any(a.uid == "isrkit-op-sentry-isr-pi" and a.source == "op-sentry" for a in assets)
        assert "OPSENTRY isr-pi" not in li.get_state()["positions"]
        assert not ots.dropped
    finally:
        sender.stop()
        li.stop()
        ots.close()


def test_kit_registration_without_position_is_ignored():
    from advisor.tak_client import build_sa

    li = CotListener()
    li._process_event(build_sa("isrkit-drone-detector-isr-pi", "DRONE-DET isr-pi", "drone-detector", None, None))
    st = li.get_state()
    assert st["positions"] == {} and st["sensor_assets"] == []
