"""LLM engine: Gemini (cloud, optional) with a local GGUF model (llama.cpp) as the offline fallback.

Backend selection, from the environment (``/etc/tac-advisor.env`` on the Pi):

``LLM_BACKEND``
    ``auto`` (default): Gemini if ``GEMINI_API_KEY`` is set, else local.
    ``gemini``: Gemini first, local model if Gemini fails or is unreachable.
    ``local``: local GGUF model only, never touches the network.
``GEMINI_API_KEY``, ``GEMINI_MODEL`` (default ``gemini-3.8-flash``)
``GEMINI_TIMEOUT_S`` (default 20)
``ADVISOR_MODELS_DIR``: where GGUF files live (default ``/var/lib/tac-advisor/models``,
falling back to ``models/`` next to the code for older installs).

After a Gemini failure the engine uses the local model for ``GEMINI_RETRY_AFTER_S``
before trying the cloud again, so an offline Pi doesn't wait for a timeout on every query.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_MODELS_DIR = Path("/var/lib/tac-advisor/models")
_LEGACY_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL = "qwen3-0.6b-q5_k_m.gguf"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_RETRY_AFTER_S = 120
_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

NO_BACKEND_MSG = (
    "No LLM available. Place a GGUF model in the models directory, or set GEMINI_API_KEY "
    "in /etc/tac-advisor.env for cloud mode."
)


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


class LLMEngine:
    """Picks a backend per request: Gemini when configured and healthy, else the local model."""

    def __init__(self, backend: str | None = None, gemini: GeminiBackend | None = None,
                 local: LocalBackend | None = None) -> None:
        self.mode = (backend or os.environ.get("LLM_BACKEND", "auto")).strip().lower()
        if self.mode not in ("auto", "gemini", "local"):
            logger.warning("Unknown LLM_BACKEND=%r, using auto", self.mode)
            self.mode = "auto"
        self._local = local or LocalBackend()
        self._gemini = gemini
        self._gemini_down_until = 0.0
        self.last_backend: str | None = None

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

    def load(self, model_name: str | None = None) -> bool:
        """Load the local model (also the offline fallback in Gemini mode)."""
        return self._local.load(model_name)

    def _gemini_usable(self) -> bool:
        return self._gemini is not None and self.mode != "local" and time.monotonic() >= self._gemini_down_until

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        if self._gemini_usable():
            try:
                text = self._gemini.generate(system_prompt, user_prompt, max_tokens)
                self.last_backend = f"gemini:{self._gemini.model_name}"
                return text
            except Exception as e:
                self._gemini_down_until = time.monotonic() + GEMINI_RETRY_AFTER_S
                logger.warning("Gemini failed (%s: %s) — using local model for %ds",
                               e.__class__.__name__, str(e)[:120], GEMINI_RETRY_AFTER_S)

        if self._local.available:
            try:
                text = self._local.generate(system_prompt, user_prompt, max_tokens)
                self.last_backend = f"local:{self._local.model_name}"
                return text
            except Exception:
                logger.exception("Local LLM generation failed")
                return "Error: LLM generation failed."
        return NO_BACKEND_MSG

    @property
    def is_loaded(self) -> bool:
        return self._local.available or self._gemini is not None

    @property
    def model_name(self) -> str | None:
        names = []
        if self._gemini is not None and self.mode != "local":
            names.append(f"gemini:{self._gemini.model_name}")
        if self._local.available:
            names.append(f"local:{self._local.model_name}")
        return " + ".join(names) or None

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "gemini": self._gemini.model_name if self._gemini is not None else None,
            "gemini_backoff_s": max(0, int(self._gemini_down_until - time.monotonic())),
            "local": self._local.model_name if self._local.available else None,
            "last_backend": self.last_backend,
        }
