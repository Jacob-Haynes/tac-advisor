"""The page uses the shared ISR kit look (ui-kit/UI-SPEC.md): links isr-ui.css/js, serves exact copies."""

import hashlib

from fastapi.testclient import TestClient

from advisor.web import app as web_module

KIT_VERSION = 3
KIT_SHA256 = {  # ui-kit v3 (isr-ui.css/js line 1 says v3)
    "isr-ui.css": "287c3df0938d25c25b04b751007095c3cf08e1df0e2689d3dd8df8b3e9eb832c",
    "isr-ui.js": "79a61cb2a427302e45b62cbebe2d022eff2dcd0549df5edea77b13a6b478fe7d",
    "fonts/OFL-BarlowCondensed.txt": "186d750eb496a4c17a76385f82be6aea2ac1cf2de074a811d63786cf374ea73f",
    "fonts/OFL-IBMPlexMono.txt": "7e6b2818edbd8f6a01ae80641cc8f16a51080d08fb4e532be3a0b6f74adb07da",
    "fonts/barlow-condensed-500.woff2": "31ff5e6e96a096fde4d78bc838719e69b093db2d0b062b96e423c065a6d1b2b1",
    "fonts/barlow-condensed-600.woff2": "5bc49d41f32810a114765314392332ce2acae2276635eee265a50319a12c1b49",
    "fonts/ibm-plex-mono-400.woff2": "52de70091213c9eab340e2a7ac59e7b6b2b94e089bf69ef38479b22cfde4274c",
    "fonts/ibm-plex-mono-500.woff2": "f17f51948d10b67d907393ccbac551912f2c2bd2c5af19be3aabe2f456f8fe54",
}


def _client():
    return TestClient(web_module.create_app())


def test_page_links_the_kit_and_has_one_header_pill():
    html = _client().get("/").text
    assert f'<link rel="stylesheet" href="/static/isr-ui.css?v={KIT_VERSION}">' in html
    assert f'<script src="/static/isr-ui.js?v={KIT_VERSION}"></script>' in html
    assert html.index("isr-ui.css") < html.index("<style>")  # kit first, own layout after
    assert html.count('id="isr-status"') == 1 and 'id="isr-kit-link"' in html
    assert html.count("isr-btn primary") <= 1
    assert "Courier New" not in html and "#00ff41" not in html  # old look gone
    page_css = html[html.index("<style>"):html.index("</style>")]
    assert "#" not in page_css.replace("#isr", "")  # no colours of our own: kit variables only
    assert ">ISR " not in html and " ISR<" not in html  # ISTAR in anything a person reads


def test_served_kit_files_are_byte_identical():
    c = _client()
    for name, sha in KIT_SHA256.items():
        r = c.get(f"/static/{name}")
        assert r.status_code == 200 and hashlib.sha256(r.content).hexdigest() == sha, name
    assert f"v{KIT_VERSION}" in c.get("/static/isr-ui.css").text.splitlines()[0]


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
