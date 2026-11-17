"""The page uses the shared ISR kit look (ui-kit/UI-SPEC.md): links isr-ui.css/js, serves exact copies."""

import hashlib

from fastapi.testclient import TestClient

from advisor.web import app as web_module

KIT_SHA256 = {
    "isr-ui.css": "4e19df9824145ead677fc30dfa28eb802294151ee16e3c4046545c6d9ce89666",
    "isr-ui.js": "a0b67a8409914a0376be7e2f51d7d4cca0fd86f7ab7ea25f048aa855cb9b1fb9",
}


def _client():
    return TestClient(web_module.create_app())


def test_page_links_the_kit_and_has_one_header_pill():
    html = _client().get("/").text
    assert '<link rel="stylesheet" href="/static/isr-ui.css">' in html
    assert '<script src="/static/isr-ui.js"></script>' in html
    assert html.index("isr-ui.css") < html.index("<style>")  # kit first, own layout after
    assert html.count('id="isr-status"') == 1 and 'id="isr-kit-link"' in html
    assert html.count("isr-btn primary") <= 1
    assert "Courier New" not in html and "#00ff41" not in html  # old look gone


def test_served_kit_files_are_byte_identical():
    c = _client()
    for name, sha in KIT_SHA256.items():
        r = c.get(f"/static/{name}")
        assert r.status_code == 200 and hashlib.sha256(r.content).hexdigest() == sha, name


def test_history_records_web_chat_and_mesh_answers():
    from advisor.history import HISTORY
    from tests.test_geochat import ALL_CHAT, OWN_CS, OWN_UID, responder
    from advisor.geochat import parse_geochat

    class Engine:
        last_backend = "ollama:qwen3.5:27b"
        is_loaded = True
        model_name = "ollama:qwen3.5:27b"

        def generate(self, system_prompt, user_prompt, max_tokens=512):
            return "SITREP: all quiet."

        def status(self):
            return {"primary": "openai_compat"}

    HISTORY.clear()
    web_module.engine, web_module.listener = Engine(), None
    c = _client()
    assert c.post("/api/query", json={"query": "Give me a SITREP"}).status_code == 200
    r, _ = responder(engine=Engine())
    r.handle(parse_geochat(ALL_CHAT, OWN_UID, OWN_CS))
    h = c.get("/api/history").json()
    assert [(e["source"], e["who"], e["query"]) for e in h] == [("chat", "HANGOVER", "SITREP"),
                                                                 ("web", "web", "Give me a SITREP")]
    assert h[0]["answer"].startswith("SITREP: all quiet") and h[0]["backend"] == "ollama:qwen3.5:27b"
    s = c.get("/api/status").json()
    assert s["simulated"] is False and s["active_backend"] == "ollama:qwen3.5:27b"
