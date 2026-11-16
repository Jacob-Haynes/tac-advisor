"""OpenAICompatBackend against fake OpenAI-compatible and Ollama servers (rehearsal TA-2)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from advisor.llm_engine import LLMEngine, OpenAICompatBackend

OPENAI_REPLY = {"id": "x", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "SITREP via v1."},
                             "finish_reason": "stop"}]}
OLLAMA_REPLY = {"model": "m", "message": {"role": "assistant", "content": "<think>x</think>SITREP: quiet."},
                "done": True}


class FakeLLMServer:
    """openai=True serves /v1/chat/completions, ollama=True serves /api/chat.
    wrong_shape_v1: /v1 answers 200 with an Ollama-shaped body (a misbehaving proxy)."""

    def __init__(self, openai=True, ollama=False, reject_think=False, wrong_shape_v1=False):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body, dict(self.headers)))
                if self.path == "/api/chat" and ollama:
                    if reject_think and "think" in body:
                        return self._send(400, {"error": "model does not support thinking"})
                    return self._send(200, OLLAMA_REPLY)
                if self.path == "/v1/chat/completions" and (openai or wrong_shape_v1):
                    return self._send(200, OLLAMA_REPLY if wrong_shape_v1 else OPENAI_REPLY)
                self._send(404, {"error": "not found"})

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def make():
    servers = []

    def _make(**kw):
        s = FakeLLMServer(**kw)
        servers.append(s)
        return s

    yield _make
    for s in servers:
        s.close()


def test_openai_shape_is_primary(make, monkeypatch):
    monkeypatch.delenv("LLM_API", raising=False)
    s = make(openai=True)
    b = OpenAICompatBackend(s.url, model="local-model", api_key="k")
    assert b.generate("sys", "user", 100) == "SITREP via v1."
    path, body, headers = s.requests[0]
    assert path == "/v1/chat/completions" and len(s.requests) == 1
    assert body["model"] == "local-model" and body["max_tokens"] == 100 and body["stream"] is False
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}]
    assert headers["Authorization"] == "Bearer k"


def test_ollama_native_when_flag_says_ollama(make):
    s = make(openai=False, ollama=True)
    b = OpenAICompatBackend(s.url, model="qwen3.5:27b", api="ollama")
    assert b.generate("sys", "user", 100) == "SITREP: quiet."
    path, body, _ = s.requests[0]
    assert path == "/api/chat" and body["think"] is False and body["stream"] is False
    assert body["options"]["num_predict"] == 100


def test_ollama_port_in_url_prefers_native():
    b = OpenAICompatBackend("http://isr-hq.local:11434")
    assert b._order() == ["ollama", "openai"]
    assert OpenAICompatBackend("http://127.0.0.1:8080")._order() == ["openai", "ollama"]
    assert OpenAICompatBackend("http://127.0.0.1:8080", api="openai")._order() == ["openai"]


def test_auto_falls_back_to_ollama_native_on_404_and_remembers(make, monkeypatch):
    monkeypatch.delenv("LLM_API", raising=False)
    s = make(openai=False, ollama=True)
    b = OpenAICompatBackend(s.url)
    assert b.generate("s", "u", 50) == "SITREP: quiet."
    assert [r[0] for r in s.requests] == ["/v1/chat/completions", "/api/chat"]
    assert b.api == "ollama"
    b.generate("s", "u", 50)
    assert s.requests[-1][0] == "/api/chat" and len(s.requests) == 3


def test_auto_falls_back_on_wrong_response_shape(make, monkeypatch):
    monkeypatch.delenv("LLM_API", raising=False)
    s = make(openai=False, ollama=True, wrong_shape_v1=True)
    assert OpenAICompatBackend(s.url).generate("s", "u", 50) == "SITREP: quiet."


def test_forced_openai_does_not_try_ollama(make):
    s = make(openai=False, ollama=True)
    b = OpenAICompatBackend(s.url, api="openai")
    with pytest.raises(Exception):
        b.generate("s", "u", 50)
    assert [r[0] for r in s.requests] == ["/v1/chat/completions"]


def test_url_with_path_is_normalised():
    assert OpenAICompatBackend("http://h:1/v1/").base_url == "http://h:1"
    assert OpenAICompatBackend("http://h:1/v1/chat/completions").base_url == "http://h:1"


def test_retries_without_think_for_models_that_reject_it(make):
    s = make(openai=False, ollama=True, reject_think=True)
    assert OpenAICompatBackend(s.url, api="ollama").generate("s", "u", 50) == "SITREP: quiet."
    assert "think" not in s.requests[-1][1]


def test_unreachable_server_falls_back_to_local():
    from tests.test_llm_engine import FakeBackend

    remote = OpenAICompatBackend("http://127.0.0.1:9", timeout_s=1)
    e = LLMEngine(backend="openai_compat", remote=remote, local=FakeBackend("l"))
    assert e.generate("s", "u") == "from l"
    assert e.status()["llm_server_backoff_s"] > 0
