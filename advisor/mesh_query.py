"""Answer ``ADVISOR:``/``@ADVISOR``-prefixed mesh texts with the LLM, replying to the sender.

Texts arrive through a mesh link (meshbus by default, see ``mesh_link``). LLM
calls take seconds, so queries are queued and answered on a worker thread
rather than in the MQTT/serial callback.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time

from advisor.context_builder import SYSTEM_PROMPT, build_context
from advisor.mesh_link import InboundText
from advisor.template_sitrep import build_sitrep, ensure_categories

logger = logging.getLogger(__name__)

ADVISOR_PREFIX = "ADVISOR:"
MAX_RESPONSE_LEN = 200
MAX_PENDING = 5
MESH_STYLE = (
    "\n\nAnswer for a LoRa mesh text: plain text, no markdown, under 200 characters."
)


_PREFIX_RE = re.compile(r"^\s*(?:ADVISOR\s*:|@ADVISOR\b[\s:,]*)", re.IGNORECASE)


def extract_query(text: str) -> str | None:
    """The query after "ADVISOR:" or "@ADVISOR" (any case), else None."""
    m = _PREFIX_RE.match(text or "")
    if not m:
        return None
    query = text[m.end():].strip()
    return query or None


def _norm(query: str) -> str:
    return " ".join(query.lower().split()).rstrip("?.! ")


class RecentQueries:
    """Answer a query once when it arrives over both the mesh and GeoChat (bridge/OTS echo).

    ``claim(channel, query, names)`` returns False if another channel claimed the same query
    from the same sender (any shared name/uid, case-insensitive) within ``window_s``.
    """

    def __init__(self, window_s: float = 30.0):
        self.window_s = window_s
        self._claims: list[tuple[float, str, str, frozenset]] = []
        self._lock = threading.Lock()

    def claim(self, channel: str, query: str, names) -> bool:
        names = frozenset(n.lower() for n in ([names] if isinstance(names, str) else names) if n)
        key, now = _norm(query), time.time()
        with self._lock:
            self._claims = [c for c in self._claims if now - c[0] < self.window_s]
            for _, ch, k, who in self._claims:
                if ch != channel and k == key and (who & names):
                    return False
            self._claims.append((now, channel, key, names))
            return True


RECENT = RecentQueries()


_ERROR_MARKERS = ("no llm available", "error:", "llm_url", "gemini_api_key", "/etc/tac-advisor.env",
                  "traceback", "llm generation failed")


def looks_like_error(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ERROR_MARKERS)


def shorten(response: str, limit: int = MAX_RESPONSE_LEN) -> str:
    response = " ".join(response.replace("*", "").replace("#", "").split())
    if len(response) > limit:
        response = response[: limit - 3] + "..."
    return response


class MeshQueryHandler:
    """Handles mesh text queries prefixed with ADVISOR:."""

    def __init__(self, engine, listener, link=None, recent: RecentQueries | None = None):
        self._recent = recent or RECENT
        self._engine = engine
        self._listener = listener
        self._link = link
        self._queue: queue.Queue[InboundText] = queue.Queue(maxsize=MAX_PENDING)
        self._running = False

    def set_link(self, link):
        self._link = link

    def start(self):
        self._running = True
        threading.Thread(target=self._worker, daemon=True, name="mesh-query").start()

    def stop(self):
        self._running = False

    def on_text(self, msg: InboundText):
        query = extract_query(msg.text)
        if query is None:
            return
        if not self._recent.claim("mesh", query, msg.names | {msg.display}):
            logger.info("Mesh query %r from %s was already answered in GeoChat", query, msg.display)
            return
        logger.info("Mesh query from %s: %s", msg.display, query)
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            logger.warning("Too many pending mesh queries, dropping one from %s", msg.from_id)
            self._reply(msg, "ADVISOR busy, try again shortly")

    def answer(self, msg: InboundText) -> str:
        """Reply text for the mesh. Never setup or error text: those are logged and the
        offline template SITREP is sent instead (rehearsal TA-1)."""
        query = extract_query(msg.text) or ""
        state = self._listener.get_state()
        try:
            context = build_context(state)
            prompt = f"{context}\n\nQuery from {msg.display}: {query}{MESH_STYLE}"
            text = shorten(self._engine.generate(SYSTEM_PROMPT, prompt, max_tokens=160), 10_000)
        except Exception:
            logger.exception("Advisor failed to answer %r", query)
            text = ""
        if not text or looks_like_error(text):
            if text:
                logger.warning("Not sending setup/error text over the mesh: %s", text)
            return shorten(build_sitrep(state, compact=True))
        if not str(getattr(self._engine, "last_backend", "") or "").startswith("template"):
            text = ensure_categories(text, state, limit=MAX_RESPONSE_LEN)  # counts can't be dropped
        return shorten(text)

    def _reply(self, msg: InboundText, text: str):
        if self._link is None:
            logger.info("Mesh reply (no link) to %s: %s", msg.from_id, text)
            return
        if self._link.send(text, to=msg.from_id, channel=msg.channel):
            logger.info("Sent response to %s (%d chars)", msg.from_id, len(text))

    def _worker(self):
        while self._running:
            try:
                msg = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                text = self.answer(msg)
                self._reply(msg, text)
                from advisor.history import HISTORY

                HISTORY.add("mesh", msg.display, extract_query(msg.text) or "", text,
                            getattr(self._engine, "last_backend", ""))
            except Exception:
                logger.exception("Failed to answer mesh query from %s", msg.from_id)  # nothing sent
