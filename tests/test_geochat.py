"""ATAK GeoChat queries: parse captured events, reply in the right shape, no loops or doubles."""

import socket
import time
import xml.etree.ElementTree as ET

import pytest

from xml.sax.saxutils import escape

from advisor.geochat import GeoChatResponder, build_reply, chat_text, parse_geochat, split_chat
from advisor.mesh_link import InboundText
from advisor.mesh_query import MeshQueryHandler, RecentQueries, extract_query

OWN_UID, OWN_CS = "isrkit-tac-advisor-isr-pi", "ADVISOR isr-pi"
PHONE = "ANDROID-5f1c2b9e0a7d4c11"

# ATAK-CIV 5.x GeoChat to "All Chat Rooms", as relayed by OTS (captured shape, ids changed)
ALL_CHAT = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<event version="2.0" uid="GeoChat.{PHONE}.All Chat Rooms.6b1d7c52-6c0e-4b8a-9d1e-2f9e5a3c7b10"
 type="b-t-f" time="2026-11-16T11:02:13.412Z" start="2026-11-16T11:02:13.412Z"
 stale="2026-11-17T11:02:13.412Z" how="h-g-i-g-o">
 <point lat="51.1504321" lon="-1.7489012" hae="112.3" ce="9.8" le="9999999.0"/>
 <detail>
  <__chat parent="RootContactGroup" groupOwner="false" messageId="6b1d7c52-6c0e-4b8a-9d1e-2f9e5a3c7b10"
   chatroom="All Chat Rooms" id="All Chat Rooms" senderCallsign="HANGOVER">
   <chatgrp uid0="{PHONE}" uid1="All Chat Rooms" id="All Chat Rooms"/>
  </__chat>
  <link uid="{PHONE}" type="a-f-G-U-C" relation="p-p"/>
  <remarks source="BAO.F.ATAK.{PHONE}" to="All Chat Rooms" time="2026-11-16T11:02:13.412Z">ADVISOR: SITREP</remarks>
  <__serverdestination destinations="192.168.43.17:4242:tcp:{PHONE}"/>
 </detail>
</event>"""

# ATAK-CIV direct message to the contact "ADVISOR isr-pi"
DM = f"""<event version="2.0" uid="GeoChat.{PHONE}.{OWN_UID}.0e7a3f1c-1b2d-4e5f-8a9b-c0d1e2f3a4b5"
 type="b-t-f" time="2026-11-16T11:05:00.000Z" start="2026-11-16T11:05:00.000Z"
 stale="2026-11-17T11:05:00.000Z" how="h-g-i-g-o">
 <point lat="51.1504321" lon="-1.7489012" hae="112.3" ce="9.8" le="9999999.0"/>
 <detail>
  <__chat parent="RootContactGroup" groupOwner="false" messageId="0e7a3f1c-1b2d-4e5f-8a9b-c0d1e2f3a4b5"
   chatroom="{OWN_CS}" id="{OWN_UID}" senderCallsign="HANGOVER">
   <chatgrp uid0="{PHONE}" uid1="{OWN_UID}" id="{OWN_UID}"/>
  </__chat>
  <link uid="{PHONE}" type="a-f-G-U-C" relation="p-p"/>
  <remarks source="BAO.F.ATAK.{PHONE}" to="{OWN_UID}" time="2026-11-16T11:05:00.000Z">@advisor any drones?</remarks>
  <__serverdestination destinations="192.168.43.17:4242:tcp:{PHONE}"/>
  <marti><dest callsign="{OWN_CS}"/></marti>
 </detail>
</event>"""

# What OTS 1.8.3's meshtastic_controller.text_message makes of a mesh text (the bridge echo)
FROM_MESH = """<event version="2.0" uid="GeoChat.!a1a1a1a1.All Chat Rooms.4c2e" type="b-t-f"
 time="2026-11-16T11:02:14Z" start="2026-11-16T11:02:14Z" stale="2026-11-17T11:02:14Z" how="h-g-i-g-o">
 <point lat="0.0" lon="0.0" hae="0" ce="9999999" le="9999999"/>
 <detail><__meshtastic meshtastic_id="2711724449"/><contact callsign="Alpha 1" endpoint="MQTT"/>
  <__chat chatroom="All Chat Rooms" groupOwner="false" id="All Chat Rooms" messageId="4c2e"
   parent="RootContactGroup" senderCallsign="Alpha 1">
   <chatgrp id="All Chat Rooms" uid0="!a1a1a1a1" uid1="All Chat Rooms"/></__chat>
  <link relation="p-p" type="a-f-G-U-C" uid="!a1a1a1a1"/>
  <remarks source="BAO.F.ATAK.!a1a1a1a1" time="2026-11-16T11:02:14Z" to="All Chat Rooms">ADVISOR: sitrep</remarks>
 </detail></event>"""

LONG = ("SITREP 1102Z. HANGOVER at grid 30U WB 87427 67248, battery 81%. 5 simulated drone detections "
        "(drone-detector), operators located 1.3 km NW of HANGOVER. 2 people/vehicle detections. "
        "No alerts. Recommend: maintain observation, report any confirmed UAV to HQ.")


class FakeEngine:
    last_backend = "ollama:qwen3.5:27b"

    def __init__(self, text=LONG):
        self.text, self.prompts = text, []

    def generate(self, system_prompt, user_prompt, max_tokens=512):
        self.prompts.append((user_prompt, max_tokens))
        return self.text


class FakeListener:
    def get_state(self):
        return {}


def responder(**kw):
    sent = []
    r = GeoChatResponder(kw.pop("engine", FakeEngine()), FakeListener(), OWN_UID, OWN_CS, send=sent.append,
                         position=lambda: (51.15, -1.75), hold_s=0, recent=kw.pop("recent", RecentQueries()))
    r.part_gap_s = 0
    return r, sent


def test_extract_query_prefixes():
    assert extract_query("ADVISOR: SITREP") == "SITREP"
    assert extract_query("advisor:sitrep") == "sitrep"
    assert extract_query("@ADVISOR sitrep") == "sitrep" and extract_query("@Advisor, any drones?") == "any drones?"
    assert extract_query("@advisorbot hi") is None and extract_query("sitrep please") is None
    assert extract_query("[HANGOVER] ADVISOR: SITREP") is None


def test_parse_all_chat_rooms_and_dm():
    m = parse_geochat(ALL_CHAT, OWN_UID, OWN_CS)
    assert (m.sender_uid, m.sender_callsign, m.chatroom, m.chat_id, m.text) == \
        (PHONE, "HANGOVER", "All Chat Rooms", "All Chat Rooms", "ADVISOR: SITREP")
    assert not m.direct and not m.from_mesh
    d = parse_geochat(DM, OWN_UID, OWN_CS)
    assert d.direct and d.text == "@advisor any drones?"
    assert parse_geochat(FROM_MESH, OWN_UID, OWN_CS).from_mesh


def ots_parse_geochat_requirements(xml):
    """What OTS 1.8.3 cot_parser.parse_geochat reads outside any try (else nack, never relayed)."""
    ev = ET.fromstring(xml)
    chat = ev.find("detail/__chat")
    assert chat.get("chatroom") and chat.get("id")
    assert chat.find("chatgrp").get("uid0") == OWN_UID
    assert ev.find("detail/remarks").get("time").endswith("Z")
    p = ev.find("point")
    assert all(p.get(k) for k in ("lat", "lon", "hae", "ce", "le"))
    return ev


def test_reply_to_all_chat_rooms_is_full_length_from_our_identity():
    r, sent = responder()
    msg = parse_geochat(ALL_CHAT, OWN_UID, OWN_CS)
    assert r.wants(msg) and r.handle(msg)
    ev = ots_parse_geochat_requirements(sent[0])
    assert ev.get("type") == "b-t-f" and ev.get("uid").startswith(f"GeoChat.{OWN_UID}.All Chat Rooms.")
    chat = ev.find("detail/__chat")
    assert (chat.get("chatroom"), chat.get("id"), chat.get("senderCallsign")) == \
        ("All Chat Rooms", "All Chat Rooms", OWN_CS)
    assert ev.find("detail/link").get("uid") == OWN_UID
    assert ev.find("detail/marti") is None  # no <dest>: OTS routes it to everyone
    assert ev.find("detail/__serverdestination").get("destinations").endswith(OWN_UID)
    assert len(LONG) > 255 and len(sent) == 2  # over OTS's geochat.remarks varchar(255): two parts
    texts = [ET.fromstring(x).find("detail/remarks").text for x in sent]
    assert texts[0].startswith("SITREP 1/2 ") and texts[1].startswith("SITREP 2/2 ")
    assert " ".join(t.split(" ", 2)[2] for t in texts) == LONG  # nothing lost, no mesh trim
    assert ev.get("stale") > ev.get("start")


def test_dm_reply_goes_back_to_the_sender():
    r, sent = responder()
    msg = parse_geochat(DM, OWN_UID, OWN_CS)
    assert r.handle(msg)
    ev = ots_parse_geochat_requirements(sent[0])
    chat = ev.find("detail/__chat")
    assert (chat.get("chatroom"), chat.get("id")) == ("HANGOVER", PHONE)
    assert chat.find("chatgrp").get("uid1") == PHONE
    assert ev.find("detail/marti/dest").get("callsign") == "HANGOVER"  # OTS dms exchange key
    assert ev.find("detail/remarks").get("to") == PHONE


def test_ignores_own_kit_unprefixed_and_mesh_converted():
    r, _ = responder()
    own = build_reply(OWN_UID, OWN_CS, parse_geochat(ALL_CHAT, OWN_UID, OWN_CS), "ADVISOR: loop bait")
    assert not r.wants(parse_geochat(own, OWN_UID, OWN_CS))
    assert not r.wants(parse_geochat(ALL_CHAT.replace("ADVISOR: SITREP", "Comms check"), OWN_UID, OWN_CS))
    assert not r.wants(parse_geochat(ALL_CHAT.replace(PHONE, "isrkit-drone-detector-isr-pi"), OWN_UID, OWN_CS))
    assert not r.wants(parse_geochat(FROM_MESH, OWN_UID, OWN_CS))


def test_same_geochat_delivered_twice_is_answered_once():
    r, _ = responder()
    msg = parse_geochat(ALL_CHAT, OWN_UID, OWN_CS)
    assert r.wants(msg) and not r.wants(parse_geochat(ALL_CHAT, OWN_UID, OWN_CS))


def test_mesh_first_then_geochat_echo_answered_once():
    recent = RecentQueries()
    mesh = MeshQueryHandler(FakeEngine(), FakeListener(), recent=recent)
    mesh.on_text(InboundText(from_id="!5f1c2b9e", text="ADVISOR: sitrep", short_name="HANG", long_name="HANGOVER"))
    assert mesh._queue.qsize() == 1
    r, sent = responder(recent=recent)
    assert not r.handle(parse_geochat(ALL_CHAT, OWN_UID, OWN_CS)) and sent == []


def test_geochat_first_then_mesh_echo_answered_once():
    recent = RecentQueries()
    r, sent = responder(recent=recent)
    assert r.handle(parse_geochat(ALL_CHAT, OWN_UID, OWN_CS)) and r.replies == 1
    mesh = MeshQueryHandler(FakeEngine(), FakeListener(), recent=recent)
    mesh.on_text(InboundText(from_id="!5f1c2b9e", text="advisor: SITREP", long_name="HANGOVER"))
    assert mesh._queue.qsize() == 0


def test_different_senders_both_answered():
    recent = RecentQueries()
    mesh = MeshQueryHandler(FakeEngine(), FakeListener(), recent=recent)
    mesh.on_text(InboundText(from_id="!a1a1a1a1", text="ADVISOR: sitrep", long_name="Alpha 1"))
    r, sent = responder(recent=recent)
    assert r.handle(parse_geochat(ALL_CHAT, OWN_UID, OWN_CS)) and r.replies == 1


def test_error_text_never_reaches_chat_and_markdown_is_stripped():
    r, sent = responder(engine=FakeEngine("Error: LLM_URL not set"))
    r.handle(parse_geochat(ALL_CHAT, OWN_UID, OWN_CS))
    assert "LLM_URL" not in sent[0] and "OFFLINE SITREP" in sent[0]
    assert chat_text("**SITREP**\n\n\n# Drones\n- 5 simulated") == "SITREP\n\nDrones\n- 5 simulated"


def test_end_to_end_through_fake_ots(monkeypatch):
    """HANGOVER (a bare ATAK-like socket) asks in All Chat Rooms; the listener answers over OTS."""
    from advisor.cot_listener import CotListener
    from tests.fake_ots import FakeOTS
    from advisor.tak_client import build_sa

    ots = FakeOTS(bind_delay=0.2)
    monkeypatch.setenv("TAK_REGISTER_DELAY_S", "0.4")
    monkeypatch.setenv("ISR_HOST", "isr-pi")
    listener = CotListener(host="127.0.0.1", port=ots.port)
    r = GeoChatResponder(FakeEngine(), listener, listener.uid, listener.callsign, send=listener.send_cot,
                         position=listener.position, hold_s=0, recent=RecentQueries()).start()
    listener.on_chat = r.on_chat
    listener.start()
    phone = socket.create_connection(("127.0.0.1", ots.port))
    try:
        assert listener._client.wait_registered(5)
        time.sleep(0.3)  # past the bind delay, like a real phone
        phone.sendall(build_sa(PHONE, "HANGOVER", "ATAK-CIV", 51.15, -1.75).encode())
        assert _until(lambda: PHONE in ots.euds)
        phone.sendall(ALL_CHAT.split("?>", 1)[1].encode())
        phone.settimeout(5)
        got = b""
        end = time.time() + 5
        while b"senderCallsign=\"ADVISOR isr-pi\"" not in got and time.time() < end:
            try:
                got += phone.recv(65536)
            except socket.timeout:
                break
        assert b'senderCallsign="ADVISOR isr-pi"' in got and b"30U WB 87427 67248" in got
        assert _until(lambda: r.sent == 2)  # LONG is two parts
        time.sleep(0.5)
        assert r.replies == 1 and r.sent == 2  # no loop on our own reply
        assert not ots.nacked and len(ots.geochats) == 3  # the query + 2 parts
    finally:
        phone.close()
        r.stop()
        listener.stop()
        ots.close()


def _until(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


SITREP_1139 = (
    "SITREP 2031Z (SIMULATED data in picture).\n"
    "1. Friendly: HANGOVER at 30U WB 87427 67248, battery 81%, active. Bravo-2 [SIMULATED] at "
    "30U WB 88112 67903, 900 m NE of HANGOVER.\n"
    "2. Drones: 5 simulated drone detections (drone-detector). UAV SIM-1581F5 at 30U WB 86790 68301, "
    "1.3 km NW of HANGOVER, alt 120 m; operator at 30U WB 86512 68420. UAV SIM-2A77 at "
    "30U WB 89001 66012, 2.0 km SE of HANGOVER. Three further tracks drifting north at low level; "
    "all labelled SIMULATED, none corroborated.\n"
    "3. Sensors: 2 people/vehicle detections (1 recorded): op-sentry OP1 reports 2xPERSON bearing 045; "
    "fpv-vision SPOT 161502 [RECORDED] SALUTE from recorded footage.\n"
    "4. Alerts: none. Messages: 3 chat messages, latest from HANGOVER.\n"
    "5. Assessment: no confirmed hostile activity. Simulated UAV pattern consistent with a "
    "reconnaissance orbit NW of HANGOVER.\n"
    "6. Recommendations: maintain OP; task acoustic sensor to the NW arc; report any live UAV to HQ "
    "immediately; confirm SIMULATED items are excluded from the real picture before handover.")


def test_1139_char_sitrep_parts_are_accepted_by_ots():
    text = (SITREP_1139 + "\n7. Next SITREP at 2100Z or on any change; all stations acknowledge "
            "receipt and report status, ammunition and water states.")[:1139]
    assert len(text) == 1139
    parts = split_chat(text)
    assert len(parts) >= 5 and all(len(escape(p)) <= 250 for p in parts)
    assert [p.split(" ", 2)[1] for p in parts] == [f"{i}/{len(parts)}" for i in range(1, len(parts) + 1)]
    assert "".join(p.split(" ", 2)[2] for p in parts).replace(" ", "").replace("\n", "") == \
        text.replace(" ", "").replace("\n", "")  # nothing lost or reordered

    from tests.fake_ots import FakeOTS

    ots = FakeOTS(bind_delay=0)
    s = socket.create_connection(("127.0.0.1", ots.port))
    try:
        from advisor.tak_client import build_sa

        s.sendall(build_sa(OWN_UID, OWN_CS, "tac-advisor", 51.15, -1.75).encode())
        msg = parse_geochat(ALL_CHAT, OWN_UID, OWN_CS)
        uids = []
        for p in parts:
            xml = build_reply(OWN_UID, OWN_CS, msg, p)
            uids.append(ET.fromstring(xml).get("uid"))
            s.sendall(xml.encode())
        assert _until(lambda: len(ots.geochats) == len(parts))
        assert not ots.nacked and len(set(uids)) == len(parts)
    finally:
        s.close()
        ots.close()


def test_fake_ots_nacks_an_unsplit_long_geochat():
    """The Pi failure: remarks over 255 -> StringDataRightTruncation -> nacked, never relayed."""
    from tests.fake_ots import FakeOTS
    from advisor.tak_client import build_sa

    ots = FakeOTS(bind_delay=0)
    s = socket.create_connection(("127.0.0.1", ots.port))
    try:
        s.sendall(build_sa(PHONE, "HANGOVER", "ATAK-CIV", 51.15, -1.75).encode())
        s.sendall(ALL_CHAT.split("?>", 1)[1].replace("ADVISOR: SITREP", "x" * 300).encode())
        assert _until(lambda: ots.nacked)
        assert "character varying(255)" in ots.nacked[0] and not ots.geochats
    finally:
        s.close()
        ots.close()


def test_split_keeps_short_answers_whole_and_cuts_unbroken_text():
    assert split_chat("SITREP: all quiet.") == ["SITREP: all quiet."]
    parts = split_chat("A" * 600 + " & <done>", label="ADVISOR")
    assert all(len(escape(p)) <= 250 for p in parts) and parts[0].startswith("ADVISOR 1/")


def test_reply_never_exceeds_ots_limit_even_unsplit():
    xml = build_reply(OWN_UID, OWN_CS, parse_geochat(ALL_CHAT, OWN_UID, OWN_CS), "y" * 400)
    assert len(ET.fromstring(xml).find("detail/remarks").text) <= 250
