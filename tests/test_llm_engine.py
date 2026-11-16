from advisor.llm_engine import NO_BACKEND_MSG, LLMEngine


class FakeBackend:
    def __init__(self, name, reply=None, fail=False, kind=None):
        self.name = kind or name
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
    g, l = FakeBackend("g", kind="gemini"), FakeBackend("l")
    e = LLMEngine(backend="auto", gemini=g, local=l)
    assert e.generate("s", "u") == "from g"
    assert e.last_backend == "gemini:g"


def test_falls_back_to_local_and_backs_off():
    g, l = FakeBackend("g", fail=True, kind="gemini"), FakeBackend("l")
    e = LLMEngine(backend="gemini", gemini=g, local=l)
    assert e.generate("s", "u") == "from l"
    assert e.generate("s", "u") == "from l"
    assert g.calls == 1  # second call skipped Gemini during the backoff
    assert e.status()["gemini_backoff_s"] > 0


def test_local_mode_never_calls_gemini():
    g, l = FakeBackend("g", kind="gemini"), FakeBackend("l")
    e = LLMEngine(backend="local", gemini=g, local=l)
    assert e.generate("s", "u") == "from l" and g.calls == 0


def test_no_backend_gives_template_sitrep_not_setup_text(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_URL", raising=False)
    l = FakeBackend("l")
    l.available = False
    e = LLMEngine(backend="auto", local=l)
    out = e.generate("s", "u")
    assert out != NO_BACKEND_MSG and "LLM_URL" not in out and out.startswith("OFFLINE SITREP")
    assert e.last_backend == "template:sitrep" and e.primary == "template"
    assert not e.is_loaded


def test_auto_without_key_has_no_gemini(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert LLMEngine(backend="auto").status()["gemini"] is None


def test_gemini_backend_builds_with_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    e = LLMEngine(backend="auto")
    assert e.status()["gemini"] == "gemini-test-model"


def test_auto_chain_order_and_explicit_mode(monkeypatch):
    r = FakeBackend("qwen3.5:27b", kind="openai_compat", fail=True)
    g, l = FakeBackend("g", kind="gemini"), FakeBackend("l")
    e = LLMEngine(backend="auto", gemini=g, local=l, remote=r)
    assert e.generate("s", "u") == "from g" and r.calls == 1  # laptop down -> gemini
    assert e.primary == "gemini"
    e2 = LLMEngine(backend="openai_compat", gemini=FakeBackend("g2", kind="gemini"), local=l,
                   remote=FakeBackend("m", kind="openai_compat", fail=True))
    assert e2.generate("s", "u") == "from l"  # explicit openai_compat never uses gemini


def test_llm_url_env_builds_backend(monkeypatch):
    monkeypatch.setenv("LLM_URL", "http://isr-hq.local:11434/v1")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    e = LLMEngine(backend="auto")
    assert e.status()["llm_server"] == "http://isr-hq.local:11434 qwen3.5:27b"
