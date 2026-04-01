"""Listen for ADVISOR: prefixed Meshtastic messages and respond via LLM."""

import logging
import threading

logger = logging.getLogger(__name__)

ADVISOR_PREFIX = "ADVISOR:"
MAX_RESPONSE_LEN = 200


class MeshQueryHandler:
    """Handles Meshtastic text queries prefixed with ADVISOR:."""

    def __init__(self, engine, listener, context_builder):
        self._engine = engine
        self._listener = listener
        self._context_builder = context_builder
        self._interface = None

    def start(self, port: str = "/dev/ttyUSB0"):
        threading.Thread(
            target=self._connect, args=(port,), daemon=True
        ).start()

    def _connect(self, port: str):
        try:
            import meshtastic.serial_interface
            from pubsub import pub

            self._interface = meshtastic.serial_interface.SerialInterface(port)
            pub.subscribe(self._on_receive, "meshtastic.receive.text")
            logger.info("Meshtastic query handler connected on %s", port)
        except Exception:
            logger.exception("Failed to connect to Meshtastic on %s", port)

    def _on_receive(self, packet, interface):
        decoded = packet.get("decoded", {})
        text = decoded.get("text", "")
        if not text:
            return

        stripped = text.strip()
        if not stripped.upper().startswith(ADVISOR_PREFIX):
            return

        query = stripped[len(ADVISOR_PREFIX):].strip()
        if not query:
            return

        from_id = str(packet.get("fromId", packet.get("from", "unknown")))
        logger.info("Query from %s: %s", from_id, query)

        # Build context and generate response
        from advisor.context_builder import SYSTEM_PROMPT, build_context

        state = self._listener.get_state()
        context = build_context(state)
        full_prompt = f"{context}\n\nQuery: {query}"

        response = self._engine.generate(SYSTEM_PROMPT, full_prompt)

        # Truncate for mesh bandwidth
        if len(response) > MAX_RESPONSE_LEN:
            response = response[:MAX_RESPONSE_LEN - 3] + "..."

        try:
            self._interface.sendText(response, destinationId=from_id)
            logger.info("Sent response to %s (%d chars)", from_id, len(response))
        except Exception:
            logger.exception("Failed to send Meshtastic response")