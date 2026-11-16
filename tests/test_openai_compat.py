"""OpenAICompatBackend against a fake Ollama / OpenAI-compatible server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from advisor.llm_engine import LLMEngine, OpenAICompatBackend


class FakeLLMServer:
    def __init__(self, ollama=True, reject_think=False):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body))
                if self.path == "/api/chat" and ollama:
                    if reject_think and "think" in body:
                        return self._send(400, {"error": "model does not support thinking"})
                    return self._send(200, {"message": {"role": "assistant",
                                                        "content": "<think>x</think>SITREP: quiet."}})
                if self.path == "/v1/chat/completions":
                    return self._send(200, {"choices": [{"message": {"content": "SITREP via v1."}}]})
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
def server():
    s = FakeLLMServer()
    yield s
    s.close()


def test_ollama_native_with_thinking_off(server):
    b = OpenAICompatBackend(server.url, model="qwen3.5:27b")
    assert b.generate("sys", "user", 100) == "SITREP: quiet."
    path, body = server.requests[0]
    assert path == "/api/chat" and body["think"] is False and body["stream"] is False
    assert body["model"] == "qwen3.5:27b" and body["messages"][0]["role"] == "system"


def test_url_with_v1_suffix_is_normalised(server):
    assert OpenAICompatBackend(server.url + "/v1/").base_url == server.url


def test_falls_back_to_openai_api_when_no_api_chat():
    s = FakeLLMServer(ollama=False)
    try:
        b = OpenAICompatBackend(s.url, model="local-model")
        assert b.generate("s", "u", 50) == "SITREP via v1."
        assert b.api == "openai"
    finally:
        s.close()


def test_retries_without_think_for_models_that_reject_it():
    s = FakeLLMServer(reject_think=True)
    try:
        assert OpenAICompatBackend(s.url).generate("s", "u", 50) == "SITREP: quiet."
        assert "think" not in s.requests[-1][1]
    finally:
        s.close()


def test_unreachable_server_falls_back_to_local():
    from tests.test_llm_engine import FakeBackend

    remote = OpenAICompatBackend("http://127.0.0.1:9", timeout_s=1)
    e = LLMEngine(backend="openai_compat", remote=remote, local=FakeBackend("l"))
    assert e.generate("s", "u") == "from l"
    assert e.status()["llm_server_backoff_s"] > 0
