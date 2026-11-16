"""LLM engine: a network LLM (Ollama on the HQ Mac, or Gemini) with a local GGUF model as fallback.

Backend selection, from the environment (``/etc/tac-advisor.env`` on the Pi):

``LLM_BACKEND``
    ``auto`` (default): every configured backend in order: openai_compat (if ``LLM_URL``
    is set), then gemini (if ``GEMINI_API_KEY`` is set), then local, then template.
    ``openai_compat``: the LLM server at ``LLM_URL``, then local, then template.
    ``gemini``: Gemini, then local, then template.
    ``local``: local GGUF model, then template; never touches the network.
    ``template``: the offline template SITREP only (no model).
    The **template** backend (``advisor.template_sitrep``) is always the last resort, so a
    query never gets a setup or error message back (it is logged instead).
``LLM_URL``: bare server base URL, no path (e.g. ``http://isr-hq.local:11434``).
    ``LLM_API`` picks the request shape:
    ``openai``: OpenAI ``POST {LLM_URL}/v1/chat/completions`` (llama.cpp server, LM Studio, vLLM, Ollama).
    ``ollama``: Ollama's native ``POST {LLM_URL}/api/chat`` with thinking off (faster for qwen3.5).
    ``auto`` (default): Ollama native if the URL says Ollama (port 11434 or "ollama" in the host),
    otherwise OpenAI; if the server answers 404/405 or with the other shape, the other API is
    tried once and remembered.
    ``LLM_MODEL`` (default ``qwen3.5:27b``), ``LLM_TIMEOUT_S`` (default 90), ``LLM_API_KEY`` (Bearer token).
``GEMINI_API_KEY``, ``GEMINI_MODEL`` (default ``gemini-3.8-flash``), ``GEMINI_TIMEOUT_S`` (default 20)
``ADVISOR_MODELS_DIR``: where GGUF files live (default ``/var/lib/tac-advisor/models``,
falling back to ``models/`` next to the code for older installs).

After a network backend fails, it is skipped for ``RETRY_AFTER_S`` before being tried
again, so a Pi whose laptop or internet is gone doesn't wait for a timeout on every query.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_MODELS_DIR = Path("/var/lib/tac-advisor/models")
_LEGACY_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL = "qwen3-0.6b-q5_k_m.gguf"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_LLM_MODEL = "qwen3.5:27b"
RETRY_AFTER_S = 120
GEMINI_RETRY_AFTER_S = RETRY_AFTER_S  # backwards-compatible name
_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

# Logged (never sent to users) when only the template SITREP is left
NO_BACKEND_MSG = (
    "No LLM available. Set LLM_URL (Ollama on the HQ Mac) or GEMINI_API_KEY in "
    "/etc/tac-advisor.env, or place a GGUF model in the models directory. Answering with the "
    "offline template SITREP."
)


class ShapeMismatch(ValueError):
    """The server answered, but not in the shape of the API we called."""


def models_dir() -> Path:
    env = os.environ.get("ADVISOR_MODELS_DIR")
    if env:
        return Path(env)
    if _DATA_MODELS_DIR.exists():
        return _DATA_MODELS_DIR
    return _LEGACY_MODELS_DIR


class LocalBackend:
    """GGUF model via llama-cpp-python, CPU only."""

    name = "local"

    def __init__(self) -> None:
        self._llm = None
        self._lock = threading.Lock()
        self.model_name: str | None = None

    @property
    def available(self) -> bool:
        return self._llm is not None

    def load(self, model_name: str | None = None) -> bool:
        directory = models_dir()
        model_path = directory / (model_name or DEFAULT_MODEL)
        if not model_path.exists():
            gguf_files = sorted(directory.glob("*.gguf")) if directory.exists() else []
            if not gguf_files:
                logger.warning("No GGUF model files found in %s — local LLM unavailable", directory)
                return False
            model_path = gguf_files[0]
            logger.info("Requested model not found, using %s", model_path.name)

        try:
            from llama_cpp import Llama
        except ImportError:
            logger.warning("llama-cpp-python not installed — local LLM unavailable")
            return False

        logger.info("Loading local LLM: %s", model_path)
        try:
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=int(os.environ.get("ADVISOR_N_CTX", "4096")),
                n_threads=int(os.environ.get("ADVISOR_N_THREADS", "3")),
                n_gpu_layers=0,
                verbose=False,
            )
            self.model_name = model_path.name
            logger.info("Local LLM loaded: %s", self.model_name)
            return True
        except Exception:
            logger.exception("Failed to load local LLM")
            return False

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        with self._lock:
            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
        text = response["choices"][0]["message"]["content"] or ""
        return _THINK_TAG_RE.sub("", text).strip()


class GeminiBackend:
    """Google Gemini via the official google-genai SDK."""

    name = "gemini"

    def __init__(self, api_key: str, model: str | None = None, timeout_s: float | None = None) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self.model_name = model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        timeout_s = timeout_s or float(os.environ.get("GEMINI_TIMEOUT_S", "20"))
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_s * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    available = True

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max(max_tokens, 1024),  # leave room for model "thinking"
                temperature=0.3,
            ),
        )
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned no text")
        return text


def _looks_like_ollama(base_url: str) -> bool:
    from urllib.parse import urlparse

    u = urlparse(base_url)
    return u.port == 11434 or "ollama" in (u.hostname or "").lower()


class OpenAICompatBackend:
    """An LLM server on the network: any OpenAI-compatible server, or Ollama's native API."""

    name = "openai_compat"
    available = True

    def __init__(self, base_url: str, model: str | None = None, timeout_s: float | None = None,
                 api: str | None = None, api_key: str | None = None) -> None:
        base = base_url.strip().rstrip("/")
        for suffix in ("/v1/chat/completions", "/api/chat", "/v1"):
            if base.endswith(suffix):  # tolerate a URL with a path; the contract is the bare base
                base = base[: -len(suffix)]
        self.base_url = base
        self.model_name = model or os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL
        self.timeout_s = timeout_s or float(os.environ.get("LLM_TIMEOUT_S", "90"))
        self.api = (api or os.environ.get("LLM_API", "auto")).strip().lower()  # auto | ollama | openai
        if self.api not in ("auto", "ollama", "openai"):
            logger.warning("Unknown LLM_API=%r, using auto", self.api)
            self.api = "auto"
        self.api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")
        self.max_tokens_cap = int(os.environ.get("LLM_MAX_TOKENS", "512"))
        self._send_think = True
        self._send_effort = True
        self._is_ollama: bool | None = None  # from GET /api/version, probed once
        self.last_api: str | None = None

    @property
    def label(self) -> str:
        """What actually answered, e.g. "ollama:qwen3.5:27b" or "openai:local-model"."""
        api = self.last_api or ("ollama" if self.api == "ollama" else "openai" if self.api == "openai" else "llm")
        return f"{api}:{self.model_name}"

    def is_ollama(self) -> bool:
        """True if the server is Ollama: the URL says so, or GET /api/version answers like Ollama."""
        if _looks_like_ollama(self.base_url):
            return True
        if self._is_ollama is None:
            try:
                req = urllib.request.Request(self.base_url + "/api/version")
                with urllib.request.urlopen(req, timeout=min(3.0, self.timeout_s)) as resp:
                    data = json.loads(resp.read() or b"{}")
                self._is_ollama = isinstance(data, dict) and "version" in data
            except urllib.error.HTTPError:
                self._is_ollama = False  # it answered, but not as Ollama
            except (OSError, ValueError):
                return False  # unreachable: decide next time
        return bool(self._is_ollama)

    def _post(self, path: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            # strict=False: Ollama can put raw control characters inside JSON strings
            # (seen in thinking text on /v1/chat/completions), which strict json.loads rejects
            return json.loads(resp.read(), strict=False)

    def _ollama(self, messages: list, max_tokens: int) -> str:
        body = {"model": self.model_name, "messages": messages, "stream": False,
                # keep the model in memory between queries (default Ollama unload is 5 min)
                "keep_alive": os.environ.get("LLM_KEEP_ALIVE", "30m"),
                "options": {"temperature": 0.3, "num_predict": min(max_tokens, self.max_tokens_cap)}}
        if self._send_think:
            body["think"] = False  # thinking models (qwen3.5) answer far faster without it
        try:
            data = self._post("/api/chat", body)
        except urllib.error.HTTPError as e:
            if e.code == 400 and self._send_think:  # model without a thinking switch
                self._send_think = False
                return self._ollama(messages, max_tokens)
            raise
        if not isinstance(data, dict) or not isinstance(data.get("message"), dict):
            raise ShapeMismatch("response is not Ollama /api/chat shape")
        return data["message"].get("content") or ""

    def _openai(self, messages: list, max_tokens: int) -> str:
        body = {"model": self.model_name, "messages": messages, "temperature": 0.3, "stream": False,
                "max_tokens": min(max_tokens, self.max_tokens_cap)}
        if self._send_effort and self.is_ollama():
            # Ollama ignores "think": false on /v1/chat/completions; reasoning_effort "none" turns
            # thinking off there (qwen3.5 otherwise thinks for over a minute before answering)
            body["reasoning_effort"] = "none"
        try:
            data = self._post("/v1/chat/completions", body)
        except urllib.error.HTTPError as e:
            if e.code == 400 and "reasoning_effort" in body:  # a version that rejects it
                self._send_effort = False
                return self._openai(messages, max_tokens)
            raise
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices or not isinstance(choices[0], dict):
            raise ShapeMismatch("response is not OpenAI chat.completions shape")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):  # some servers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content or choices[0].get("text") or ""

    def _order(self) -> list[str]:
        if self.api in ("openai", "ollama"):
            return [self.api]
        # auto: Ollama's native /api/chat (thinking reliably off) whenever the server is Ollama
        first = "ollama" if self.is_ollama() else "openai"
        return [first, "openai" if first == "ollama" else "ollama"]

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        order = self._order()
        text = ""
        for i, api in enumerate(order):
            try:
                text = self._openai(messages, max_tokens) if api == "openai" else self._ollama(messages, max_tokens)
            except (urllib.error.HTTPError, ShapeMismatch) as e:
                code = getattr(e, "code", None)
                if i + 1 < len(order) and (isinstance(e, ShapeMismatch) or code in (404, 405)):
                    logger.info("%s: %s API not usable (%s), trying %s", self.base_url, api,
                                code or e, order[i + 1])
                    continue
                raise
            if self.api == "auto" and i > 0:
                self.api = api  # remember what worked
            self.last_api = api
            break
        text = _THINK_TAG_RE.sub("", text).strip()
        if not text:
            raise RuntimeError("LLM server returned no text")
        return text


class LLMEngine:
    """Tries network backends in order (each with its own backoff), then the local model."""

    def __init__(self, backend: str | None = None, gemini: GeminiBackend | None = None,
                 local: LocalBackend | None = None, remote: OpenAICompatBackend | None = None,
                 state_fn=None) -> None:
        from advisor.template_sitrep import TemplateBackend

        self.mode = (backend or os.environ.get("LLM_BACKEND", "auto")).strip().lower()
        force_api = None
        if self.mode in ("ollama", "openai"):  # the LLM_URL server, with that request shape
            force_api, self.mode = self.mode, "openai_compat"
        if self.mode not in ("auto", "openai_compat", "gemini", "local", "template"):
            logger.warning("Unknown LLM_BACKEND=%r, using auto", self.mode)
            self.mode = "auto"
        self._local = local or LocalBackend()
        self._gemini = gemini
        self._remote = remote
        self._template = TemplateBackend(state_fn)
        self._warned_template = False
        self._down_until: dict[str, float] = {}
        self.last_backend: str | None = None

        if self._remote is None and self.mode in ("auto", "openai_compat"):
            url = os.environ.get("LLM_URL", "").strip()
            if url:
                self._remote = OpenAICompatBackend(url, api=force_api)
                logger.info("LLM server backend: %s model %s", self._remote.base_url, self._remote.model_name)
            elif self.mode == "openai_compat":
                logger.warning("LLM_BACKEND=openai_compat but LLM_URL is not set — using local model")

        if self._gemini is None and self.mode in ("auto", "gemini"):
            key = os.environ.get("GEMINI_API_KEY", "").strip()
            if key:
                try:
                    self._gemini = GeminiBackend(key)
                    logger.info("Gemini backend enabled (model %s)", self._gemini.model_name)
                except Exception:
                    logger.exception("Gemini backend unavailable (google-genai missing?)")
            elif self.mode == "gemini":
                logger.warning("LLM_BACKEND=gemini but GEMINI_API_KEY is not set — using local model")

    def set_state_provider(self, state_fn) -> None:
        """Where the template SITREP reads the picture from (CotListener.get_state)."""
        self._template.state_fn = state_fn

    def _chain(self) -> list:
        if self.mode in ("local", "template"):
            return []
        chain = []
        if self.mode in ("auto", "openai_compat") and self._remote is not None:
            chain.append(self._remote)
        if self.mode in ("auto", "gemini") and self._gemini is not None:
            chain.append(self._gemini)
        return chain

    def load(self, model_name: str | None = None) -> bool:
        """Load the local model (also the offline fallback for the network backends)."""
        if self.mode == "template":
            return False
        return self._local.load(model_name)

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        for backend in self._chain():
            if time.monotonic() < self._down_until.get(backend.name, 0.0):
                continue
            try:
                text = backend.generate(system_prompt, user_prompt, max_tokens)
                self.last_backend = getattr(backend, "label", None) or f"{backend.name}:{backend.model_name}"
                return text
            except Exception as e:
                self._down_until[backend.name] = time.monotonic() + RETRY_AFTER_S
                logger.warning("%s failed (%s: %s) — skipping it for %ds", backend.name,
                               e.__class__.__name__, str(e)[:120], RETRY_AFTER_S)

        if self._local.available and self.mode != "template":
            try:
                text = self._local.generate(system_prompt, user_prompt, max_tokens)
                if text:
                    self.last_backend = f"local:{self._local.model_name}"
                    return text
                logger.warning("Local LLM returned no text")
            except Exception:
                logger.exception("Local LLM generation failed")

        # Last resort: a factual summary from the picture. Errors and setup hints stay in the log.
        if not self._warned_template and self.mode != "template":
            logger.warning(NO_BACKEND_MSG)
            self._warned_template = True
        self.last_backend = f"{self._template.name}:{self._template.model_name}"
        return self._template.generate(system_prompt, user_prompt, max_tokens)

    @property
    def is_loaded(self) -> bool:
        """True if a real model (not only the template) can answer."""
        return self._local.available or bool(self._chain())

    @property
    def model_name(self) -> str | None:
        names = [f"{b.name}:{b.model_name}" for b in self._chain()]
        if self._local.available and self.mode != "template":
            names.append(f"local:{self._local.model_name}")
        names.append("template:sitrep")
        return " + ".join(names)

    def backoff_s(self, name: str) -> int:
        return max(0, int(self._down_until.get(name, 0.0) - time.monotonic()))

    @property
    def primary(self) -> str:
        """Which backend should be answering right now: openai_compat | gemini | local | template."""
        for b in self._chain():
            if not self.backoff_s(b.name):
                return b.name
        return "local" if self._local.available and self.mode != "template" else "template"

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "primary": self.primary,
            "llm_server": f"{self._remote.base_url} {self._remote.model_name}" if self._remote else None,
            "llm_server_backoff_s": self.backoff_s("openai_compat"),
            "gemini": self._gemini.model_name if self._gemini is not None else None,
            "gemini_backoff_s": self.backoff_s("gemini"),
            "local": self._local.model_name if self._local.available else None,
            "template": True,
            "last_backend": self.last_backend,
        }
