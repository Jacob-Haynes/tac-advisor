import time

from advisor.cot_listener import CotListener
from advisor.mesh_link import InboundText, parse_rx
from advisor.mesh_query import MeshQueryHandler, extract_query, shorten


class FakeEngine:
    def __init__(self):
        self.prompts = []

    def generate(self, system, user, max_tokens=512):
        self.prompts.append(user)
        return "**SITREP**: all quiet. " + "x" * 400


class FakeLink:
    def __init__(self):
        self.sent = []

    def send(self, text, to=None, channel=0):
        self.sent.append((to, channel, text))
        return True


def test_extract_and_shorten():
    assert extract_query("advisor: sitrep") == "sitrep"
    assert extract_query("ADVISOR:") is None
    assert extract_query("hello") is None
    out = shorten("**bold** " + "y" * 500)
    assert len(out) == 200 and "*" not in out


def test_query_answered_to_sender_on_worker():
    engine, link = FakeEngine(), FakeLink()
    h = MeshQueryHandler(engine, CotListener(), link)
    h.start()
    try:
        h.on_text(InboundText(from_id="!a1a1a1a1", text="ADVISOR: sitrep", channel=1, short_name="A1"))
        h.on_text(InboundText(from_id="!a1a1a1a1", text="just chatting"))
        end = time.time() + 3
        while not link.sent and time.time() < end:
            time.sleep(0.02)
        assert len(link.sent) == 1
        to, ch, text = link.sent[0]
        assert to == "!a1a1a1a1" and ch == 1 and len(text) <= 200
        assert "Query from A1: sitrep" in engine.prompts[0]
    finally:
        h.stop()


def test_parse_rx_meshbus_contract():
    m = parse_rx(b'{"from": "!a1a1a1a1", "text": "ADVISOR: hi", "channel": 0, "ts": 1}')
    assert m.from_id == "!a1a1a1a1" and m.display == "!a1a1a1a1"
