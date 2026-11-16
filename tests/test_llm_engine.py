from advisor.llm_engine import NO_BACKEND_MSG, LLMEngine


class FakeBackend:
    def __init__(self, name, reply=None, fail=False):
        self.model_name = name
        self.reply = reply or f"from {name}"
        self.fail = fail
        self.calls = 0
        self.available = True

    def generate(self, system, user, max_tokens):
        self.calls += 1
        if self.fail:
            raise ConnectionError("offline")
        return self.reply


def test_gemini_used_when_configured():
    g, l = FakeBackend("g"), FakeBackend("l")
    e = LLMEngine(backend="auto", gemini=g, local=l)
    assert e.generate("s", "u") == "from g"
    assert e.last_backend == "gemini:g"


def test_falls_back_to_local_and_backs_off():
    g, l = FakeBackend("g", fail=True), FakeBackend("l")
    e = LLMEngine(backend="gemini", gemini=g, local=l)
    assert e.generate("s", "u") == "from l"
    assert e.generate("s", "u") == "from l"
    assert g.calls == 1  # second call skipped Gemini during the backoff
    assert e.status()["gemini_backoff_s"] > 0


def test_local_mode_never_calls_gemini():
    g, l = FakeBackend("g"), FakeBackend("l")
    e = LLMEngine(backend="local", gemini=g, local=l)
    assert e.generate("s", "u") == "from l" and g.calls == 0


def test_no_backend_message(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    l = FakeBackend("l")
    l.available = False
    e = LLMEngine(backend="auto", local=l)
    assert e.generate("s", "u") == NO_BACKEND_MSG
    assert not e.is_loaded


def test_auto_without_key_has_no_gemini(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert LLMEngine(backend="auto").status()["gemini"] is None


def test_gemini_backend_builds_with_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    e = LLMEngine(backend="auto")
    assert e.status()["gemini"] == "gemini-test-model"
