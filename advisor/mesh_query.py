"""Answer ``ADVISOR:``-prefixed mesh texts with the LLM, replying to the sender.

Texts arrive through a mesh link (meshbus by default, see ``mesh_link``). LLM
calls take seconds, so queries are queued and answered on a worker thread
rather than in the MQTT/serial callback.
"""

from __future__ import annotations

import logging
import queue
import threading

from advisor.context_builder import SYSTEM_PROMPT, build_context
from advisor.mesh_link import InboundText

logger = logging.getLogger(__name__)

ADVISOR_PREFIX = "ADVISOR:"
MAX_RESPONSE_LEN = 200
MAX_PENDING = 5
MESH_STYLE = (
    "\n\nAnswer for a LoRa mesh text: plain text, no markdown, under 200 characters."
)


def extract_query(text: str) -> str | None:
    stripped = text.strip()
    if not stripped.upper().startswith(ADVISOR_PREFIX):
        return None
    query = stripped[len(ADVISOR_PREFIX):].strip()
    return query or None


def shorten(response: str, limit: int = MAX_RESPONSE_LEN) -> str:
    response = " ".join(response.replace("*", "").replace("#", "").split())
    if len(response) > limit:
        response = response[: limit - 3] + "..."
    return response


class MeshQueryHandler:
    """Handles mesh text queries prefixed with ADVISOR:."""

    def __init__(self, engine, listener, link=None):
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
        logger.info("Mesh query from %s: %s", msg.display, query)
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            logger.warning("Too many pending mesh queries, dropping one from %s", msg.from_id)
            self._reply(msg, "ADVISOR busy, try again shortly")

    def answer(self, msg: InboundText) -> str:
        query = extract_query(msg.text) or ""
        context = build_context(self._listener.get_state())
        prompt = f"{context}\n\nQuery from {msg.display}: {query}{MESH_STYLE}"
        return shorten(self._engine.generate(SYSTEM_PROMPT, prompt, max_tokens=160))

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
                self._reply(msg, self.answer(msg))
            except Exception:
                logger.exception("Failed to answer mesh query from %s", msg.from_id)
