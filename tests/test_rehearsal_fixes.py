"""Rehearsal fixes: TA-1 no setup/error text on the mesh, TA-3 template SITREP, SIM/REC labels (C4.3)."""

import time

from advisor.context_builder import build_context
from advisor.cot_listener import CotListener, SimulatedCotListener, data_label
from advisor.llm_engine import NO_BACKEND_MSG, LLMEngine
from advisor.mesh_link import InboundText
from advisor.mesh_query import MeshQueryHandler, looks_like_error
from advisor.template_sitrep import build_sitrep, count_phrase


def cot(uid, cot_type, callsign, remarks="", lat=51.15, lon=-1.75):
    t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return (f'<event version="2.0" uid="{uid}" type="{cot_type}" how="m-g" time="{t}" start="{t}" stale="{t}">'
            f'<point lat="{lat}" lon="{lon}" hae="0" ce="10" le="10"/><detail><contact callsign="{callsign}"/>'
            f'<remarks>{remarks}</remarks></detail></event>')


def picture():
    li = CotListener()
    for x in (
        cot("DRONE-SIM-AC1234", "a-h-A-M-F-Q", "SIM-1581F5", "SIMULATED: drone-detector: DRONE_CONFIRMED"),
        cot("DRONE-SIM-BB0001", "a-u-A-M-F-Q", "SIM-UAV 00:01", "SIMULATED: drone-detector: DRONE_PROBABLE"),
        cot("fpv-vision-spot-1", "b-m-p-s-p-i", "SPOT 161502", "RECORDED: SALUTE from recorded footage"),
        cot("opsentry-OP1-7", "a-u-G-U-C-I", "OP1 2xPERSON BRG 045", "op-sentry: 2 person(s)"),
        cot("ANDROID-1", "a-f-G-U-C", "Alpha-1"),
        cot("ANDROID-2", "a-f-G-U-C", "Bravo-2", "SIMULATED: BFT position"),
    ):
        li._process_event(x)
    return li


class FakeLink:
    def __init__(self):
        self.sent = []

    def send(self, text, to=None, channel=0):
        self.sent.append(text)
        return True


class Backend:
    available = True

    def __init__(self, reply="", exc=None):
        self.name, self.model_name, self.reply, self.exc = "local", "m", reply, exc

    def generate(self, *a):
        if self.exc:
            raise self.exc
        return self.reply


# ── labels ───────────────────────────────────────────────────────

def test_data_label_rules():
    assert data_label(remarks="SIMULATED: RF link") == "SIMULATED"
    assert data_label(remarks="[SIM] OP1 ...") == "SIMULATED"
    assert data_label(remarks="RECORDED: SALUTE") == "RECORDED"
    assert data_label(callsign="SIM-UAV 22:41") == "SIMULATED"
    assert data_label(uid="DRONE-SIM-AC12") == "SIMULATED"
    assert data_label(uid="DRONE-AC12", callsign="1581F5", remarks="drone-detector: confirmed") == ""


def test_count_phrase():
    class I:
        def __init__(self, label=""):
            self.label = label
    assert count_phrase([I("SIMULATED"), I("SIMULATED")], "drone detection") == "2 simulated drone detections"
    assert count_phrase([I()], "drone detection") == "1 drone detection"
    assert count_phrase([I("SIMULATED"), I("RECORDED"), I()], "sensor detection") == \
        "3 sensor detections (1 simulated, 1 recorded)"


def test_context_tags_labels_for_the_llm():
    ctx = build_context(picture().get_state())
    assert "SIM-1581F5 [drone-detector] [SIMULATED]" in ctx
    assert "SPOT 161502 [RECORDED]" in ctx
    assert "Bravo-2 [SIMULATED]" in ctx and "Alpha-1:" in ctx
    assert ctx.startswith("PICTURE COUNTS") and "\n\nDATA LABELS:" in ctx
    assert "2 simulated drone detections" in ctx


def test_simulated_listener_labels_everything():
    li = SimulatedCotListener()
    li._running = True
    import threading
    t = threading.Thread(target=li._simulate_loop, daemon=True)
    t.start()
    time.sleep(0.3)
    li._running = False
    st = li.get_state()
    assert st["positions"] and all(p.label == "SIMULATED" for p in st["positions"].values())


# ── TA-3 template SITREP ─────────────────────────────────────────

def test_template_sitrep_full_and_compact():
    st = picture().get_state()
    full = build_sitrep(st)
    assert full.startswith("OFFLINE SITREP") and "2 simulated drone detections" in full
    assert "drone-detector 2" in full and "Units: 2 units (1 simulated)" in full
    assert "exercise data" in full
    short = build_sitrep(st, compact=True)
    assert len(short) <= 200 and "(offline, no LLM)" in short and "2 simulated drone detections" in short
    assert "2 sensor detections (1 recorded)" in short
    empty = build_sitrep({}, compact=True)
    assert "no picture received" in empty and "last never" in empty


def test_template_is_last_resort_in_auto(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_URL", raising=False)
    li = picture()
    local = Backend(exc=RuntimeError("boom"))
    e = LLMEngine(backend="auto", local=local, state_fn=li.get_state)
    out = e.generate("s", "u", max_tokens=160)
    assert out.startswith("SITREP ") and "2 simulated drone detections" in out
    assert e.last_backend == "template:sitrep"
    assert LLMEngine(backend="template", local=Backend("x")).generate("s", "u").startswith("OFFLINE SITREP")


# ── TA-1 never setup/error text on the mesh ──────────────────────

def test_mesh_never_gets_setup_text(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_URL", raising=False)
    li, link = picture(), FakeLink()
    unloaded = Backend()
    unloaded.available = False
    engine = LLMEngine(backend="auto", local=unloaded)
    engine.set_state_provider(li.get_state)
    h = MeshQueryHandler(engine, li, link)
    reply = h.answer(InboundText(from_id="!a", text="ADVISOR: sitrep"))
    assert not looks_like_error(reply) and "LLM_URL" not in reply and reply.startswith("SITREP")


def test_mesh_error_text_from_a_backend_is_replaced(caplog):
    class ErrEngine:
        def generate(self, *a, **k):
            return NO_BACKEND_MSG

    class Boom:
        def generate(self, *a, **k):
            raise RuntimeError("down")

    li = picture()
    for eng in (ErrEngine(), Boom()):
        reply = MeshQueryHandler(eng, li, FakeLink()).answer(InboundText(from_id="!a", text="ADVISOR: sitrep"))
        assert reply.startswith("SITREP") and "simulated" in reply and len(reply) <= 200



# ── rehearsal 3: an LLM must not drop a category ─────────────────

def busy_picture():
    li = picture()
    for i in range(3):
        li._process_event(cot(f"DRONE-SIM-X{i}", "a-h-A-M-F-Q", f"SIM-UAV {i}", "SIMULATED: drone-detector"))
    li._process_event(cot("RFSPEC-433", "a-u-G", "RF 433.920", "SIMULATED: RF: new emitter"))
    li._process_event(cot("ACOUSTIC-OP1-UAV", "a-u-A-M-F-Q", "UAV? (acoustic OP1)", "acoustic-detector"))
    li._process_event(cot("change-detect-a1", "a-u-G", "CHANGE a1 (2)", "change-detect: new tracks"))
    return li


def test_category_counts_cover_every_nonzero_category():
    from advisor.template_sitrep import category_counts

    rows = {k: phrase for k, _, phrase in category_counts(busy_picture().get_state())}
    assert rows["drones"] == "6 drone detections (5 simulated)"
    assert set(rows) == {"drones", "people", "rf", "acoustic", "change", "positions"}
    assert rows["people"] == "2 people/vehicle detections (1 recorded)"


class DropsDrones:
    """A model that answers fluently but leaves out the drones (as qwen3.5:27b did)."""
    last_backend = "ollama:qwen3.5:27b"

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def generate(self, system, user, max_tokens=512):
        self.prompts.append((system, user))
        return self.reply


def test_every_nonzero_category_appears_in_mesh_and_web_answers():
    from advisor.template_sitrep import category_counts, ensure_categories, missing_categories

    li = busy_picture()
    st = li.get_state()
    eng = DropsDrones("SITREP: 2 units in position, OP1 reports 2 people. All quiet otherwise.")
    reply = MeshQueryHandler(eng, li, FakeLink()).answer(InboundText(from_id="!a", text="ADVISOR: sitrep"))
    assert len(reply) <= 200
    for _, _, phrase in category_counts(st):
        assert phrase in reply or not missing_categories(reply, st), phrase
    assert "6 drone detections (5 simulated)" in reply
    # the prompt carried the counts and the instruction
    system, user = eng.prompts[0]
    assert "PICTURE COUNTS" in user and "6 drone detections (5 simulated)" in user
    assert "never leave a category out" in system
    full = ensure_categories(eng.reply, st)
    assert not missing_categories(full, st) and "1 simulated RF detection" in full


def test_complete_answer_is_left_alone():
    from advisor.template_sitrep import ensure_categories

    st = busy_picture().get_state()
    text = ("6 drone detections (5 simulated), 2 people seen, 1 RF emitter, 1 acoustic UAV, "
            "1 change detection, 2 unit positions.")
    assert ensure_categories(text, st) == text
