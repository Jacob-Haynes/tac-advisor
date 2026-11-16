from fastapi.testclient import TestClient

from advisor.cot_listener import CotListener
from advisor.llm_engine import LLMEngine
from advisor.web import app as web_module
from tests.test_cot_listener import OPSENTRY_PERSON
from tests.test_llm_engine import FakeBackend


def test_query_status_and_context():
    li = CotListener()
    li._process_event(OPSENTRY_PERSON)
    web_module.listener = li
    web_module.engine = LLMEngine(backend="local", local=FakeBackend("l", reply="Two pax at OP1."))
    client = TestClient(web_module.create_app())
    r = client.post("/api/query", json={"query": "sitrep"}).json()
    assert r == {"response": "Two pax at OP1.", "backend": "local:l"}
    s = client.get("/api/status").json()
    assert s["sensor_events"] == 1 and s["llm"]["mode"] == "local"
    assert "op-sentry" in client.get("/api/context").json()["context"]
