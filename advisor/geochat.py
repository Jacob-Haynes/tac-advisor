"""Answer ``ADVISOR:`` queries sent from ATAK GeoChat, replying in the same chat.

tac-advisor is a registered TAK client (contract v5), so GeoChat that OpenTAKServer relays
reaches it: "All Chat Rooms" messages (no ``<dest>``, routed to everyone) and direct messages to
the contact "ADVISOR <host>" (``<marti><dest callsign="ADVISOR <host>"/>``, routed by OTS to the
queue bound to our callsign).

Inbound (ATAK-CIV shape)::

    <event uid="GeoChat.<sender uid>.<chat id>.<messageId>" type="b-t-f" how="h-g-i-g-o" ...>
      <point .../>
      <detail>
        <__chat parent="RootContactGroup" groupOwner="false" messageId="<id>"
                chatroom="All Chat Rooms" id="All Chat Rooms" senderCallsign="HANGOVER">
          <chatgrp uid0="<sender uid>" uid1="All Chat Rooms" id="All Chat Rooms"/>
        </__chat>
        <link uid="<sender uid>" type="a-f-G-U-C" relation="p-p"/>
        <remarks source="BAO.F.ATAK.<sender uid>" to="All Chat Rooms" time="...">ADVISOR: SITREP</remarks>
        <__serverdestination destinations="<ip>:4242:tcp:<sender uid>"/>
      </detail>
    </event>

For a DM, ``chatroom`` is our callsign, ``id``/``uid1`` our uid, and there is a
``<marti><dest callsign="ADVISOR <host>"/></marti>``.

The reply uses the same shape from our registered uid/callsign: to "All Chat Rooms" with no
``<dest>``, or, for a DM, to ``chatroom``=sender callsign, ``id``=sender uid with
``<marti><dest callsign=sender>``. OTS 1.8.3's ``parse_geochat`` runs before routing and outside
any try for ``chatgrp@uid0``, ``__chat@id`` and ``remarks@time``: a reply missing any of them
is nacked and never relayed, so all three are always present.

Loops and duplicates: our own messages, other kit tools, and anything without the
``ADVISOR:``/``@ADVISOR`` prefix are ignored. GeoChat that OTS made from a mesh text
(``<contact endpoint="MQTT">``) is left to the mesh handler, which answers over the radio. Any
query also seen on the mesh shortly before or after (the bridge echo) is answered once only.
GeoChat replies carry the full SITREP (no 200-character mesh limit), but OTS 1.8.3 stores
``geochat.remarks`` as ``varchar(255)`` and nacks (never relays) a longer one, so a long answer
is sent as several messages, "SITREP 1/5 ...", each with remarks of at most 250 characters
after XML escaping, split on line, then sentence, then word boundaries, about 300 ms apart.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from xml.sax.saxutils import escape, quoteattr

from advisor.context_builder import SYSTEM_PROMPT, build_context
from advisor.mesh_query import RECENT, extract_query, looks_like_error
from advisor.tak_client import OTS_FIELD_MAX, ots_fit
from advisor.template_sitrep import build_sitrep, ensure_categories

logger = logging.getLogger(__name__)

ALL_CHAT = "All Chat Rooms"
KIT_UID_PREFIX = "isrkit-"
HOLD_S = 3.0  # wait for a mesh copy of the same query before answering here
MAX_PENDING = 5
CHAT_STALE_S = 24 * 3600
PART_GAP_S = 0.3
REMARKS_MAX = OTS_FIELD_MAX  # 250: under OTS's geochat.remarks varchar(255)
CHAT_STYLE = ("\n\nAnswer for an ATAK chat message: plain text, no markdown, short lines; "
              "a full SITREP is fine.")


@dataclass
class GeoChatMsg:
    uid: str
    sender_uid: str
    sender_callsign: str
    chatroom: str
    chat_id: str
    text: str
    direct: bool = False  # a DM to us
    from_mesh: bool = False  # OTS made it from a Meshtastic text
    received: float = field(default_factory=time.time)


def parse_geochat(xml: str | ET.Element, own_uid: str = "", own_callsign: str = "") -> GeoChatMsg | None:
    root = ET.fromstring(xml) if isinstance(xml, str) else xml
    if root.get("type") != "b-t-f":
        return None
    detail = root.find("detail")
    chat = detail.find("__chat") if detail is not None else None
    if chat is None:
        return None
    grp = chat.find("chatgrp")
    remarks = detail.find("remarks")
    link = detail.find("link")
    sender_uid = (grp.get("uid0") if grp is not None else "") or (link.get("uid") if link is not None else "")
    if not sender_uid:
        parts = root.get("uid", "").split(".")
        sender_uid = parts[1] if len(parts) > 2 and parts[0] == "GeoChat" else ""
    chatroom = chat.get("chatroom", ALL_CHAT)
    chat_id = chat.get("id", chatroom)
    contact = detail.find("contact")
    direct = bool(own_uid) and (chat_id == own_uid or (grp is not None and grp.get("uid1") == own_uid)
                                or (own_callsign and chatroom == own_callsign))
    return GeoChatMsg(
        uid=root.get("uid", ""), sender_uid=sender_uid,
        sender_callsign=chat.get("senderCallsign", "") or sender_uid,
        chatroom=chatroom, chat_id=chat_id, text=(remarks.text or "") if remarks is not None else "",
        direct=direct,
        from_mesh=(contact is not None and contact.get("endpoint") == "MQTT")
        or detail.find("__meshtastic") is not None,
    )


def _iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t % 1 * 1000):03d}Z"


def _xml_len(text: str) -> int:
    return len(escape(text))


def _pieces(text: str, room: int):
    """(separator, piece) units no longer than ``room`` (escaped): lines, else sentences,
    else words, else hard cuts."""
    for line in text.split("\n"):
        if _xml_len(line) <= room:
            yield "\n", line
            continue
        sep = "\n"
        for sentence in re.split(r"(?<=[.!?;])\s+", line):
            if _xml_len(sentence) <= room:
                yield sep, sentence
            else:
                for word in sentence.split(" "):
                    while _xml_len(word) > room:
                        cut = room
                        while _xml_len(word[:cut]) > room:
                            cut -= 1
                        yield sep, word[:cut]
                        sep, word = " ", word[cut:]
                    yield sep, word
                    sep = " "
            sep = " "


def split_chat(text: str, label: str = "SITREP", limit: int = REMARKS_MAX) -> list[str]:
    """``text`` as one message if it fits in ``limit`` escaped characters, else as parts
    "LABEL 1/n ..." that each do."""
    text = text.strip()
    if _xml_len(text) <= limit:
        return [text]
    prefix_room = len(f"{label} 99/99 ")
    room = limit - prefix_room
    parts, current = [], ""
    for sep, piece in _pieces(text, room):
        if not piece and sep == " ":
            continue
        candidate = current + sep + piece if current else piece
        if _xml_len(candidate) <= room:
            current = candidate
        else:
            if current.strip():
                parts.append(current.strip())
            current = piece
    if current.strip():
        parts.append(current.strip())
    n = len(parts)
    return [f"{label} {i}/{n} {p}" for i, p in enumerate(parts, 1)]


def reply_label(query: str) -> str:
    return "SITREP" if "sitrep" in query.lower() else "ADVISOR"


def build_reply(own_uid: str, own_callsign: str, to: GeoChatMsg, text: str,
                lat: float | None = None, lon: float | None = None, now: float | None = None) -> str:
    """GeoChat CoT from us: to the same chat room, or a DM back to the sender."""
    now = time.time() if now is None else now
    msg_id = str(uuid.uuid4())
    text = ots_fit(text, "GeoChat remarks")  # split_chat keeps parts under this; last-resort cut
    if to.direct:
        room, room_id = to.sender_callsign, to.sender_uid
        dest = f'<marti><dest callsign={quoteattr(to.sender_callsign)}/></marti>'
    else:
        room, room_id = to.chatroom, to.chat_id
        dest = ""
    if lat is not None and lon is not None and (lat or lon):
        point = f'<point lat="{lat:.7f}" lon="{lon:.7f}" hae="9999999.0" ce="9999999.0" le="9999999.0"/>'
    else:
        point = '<point lat="0" lon="0" hae="9999999.0" ce="9999999.0" le="9999999.0"/>'
    room, room_id = ots_fit(room, "GeoChat chatroom"), ots_fit(room_id, "GeoChat chatroom id")
    for value in (own_uid, own_callsign, room, room_id, f"GeoChat.{own_uid}.{room_id}.{msg_id}"):
        assert len(value) <= 255, f"GeoChat field over OTS varchar(255): {value[:40]}..."
    t = _iso(now)
    return (
        f'<event version="2.0" uid={quoteattr(f"GeoChat.{own_uid}.{room_id}.{msg_id}")} type="b-t-f" '
        f'how="h-g-i-g-o" time="{t}" start="{t}" stale="{_iso(now + CHAT_STALE_S)}">{point}<detail>'
        f'<__chat parent="RootContactGroup" groupOwner="false" messageId="{msg_id}" '
        f'chatroom={quoteattr(room)} id={quoteattr(room_id)} senderCallsign={quoteattr(own_callsign)}>'
        f'<chatgrp uid0={quoteattr(own_uid)} uid1={quoteattr(room_id)} id={quoteattr(room_id)}/></__chat>'
        f'<link uid={quoteattr(own_uid)} type="a-f-G-E-S" relation="p-p"/>'
        f'<remarks source={quoteattr("BAO.F.ATAK." + own_uid)} to={quoteattr(room_id)} time="{t}">'
        f'{escape(text)}</remarks>'
        f'<__serverdestination destinations={quoteattr("*:-1:stcp:" + own_uid)}/>'
        f'{dest}</detail></event>'
    )


def chat_text(text: str) -> str:
    """Markdown off, line breaks kept (ATAK shows multi-line chat)."""
    lines = [" ".join(line.replace("*", "").replace("#", "").split()) for line in text.splitlines()]
    out, blank = [], False
    for line in lines:
        if not line:
            if out and not blank:
                out.append("")
            blank = True
            continue
        out.append(line)
        blank = False
    return "\n".join(out).strip()


class GeoChatResponder:
    """Queues GeoChat ADVISOR queries and answers them on a worker thread via ``send(xml)``."""

    def __init__(self, engine, listener, own_uid: str, own_callsign: str, send, position=None,
                 hold_s: float = HOLD_S, recent=RECENT):
        self._engine = engine
        self._listener = listener
        self.own_uid = own_uid
        self.own_callsign = own_callsign
        self._send = send
        self._position = position or (lambda: (None, None))
        self._hold_s = hold_s
        self._recent = recent
        self._queue: queue.Queue[GeoChatMsg] = queue.Queue(maxsize=MAX_PENDING)
        self._seen: dict[str, float] = {}
        self._running = False
        self.sent = 0  # GeoChat messages sent
        self.replies = 0  # queries answered (a reply may be several messages)
        self.part_gap_s = PART_GAP_S

    def start(self):
        self._running = True
        threading.Thread(target=self._worker, daemon=True, name="geochat-query").start()
        return self

    def stop(self):
        self._running = False

    def wants(self, msg: GeoChatMsg) -> bool:
        if msg.sender_uid == self.own_uid or msg.sender_callsign == self.own_callsign:
            return False  # our own reply
        if msg.sender_uid.startswith(KIT_UID_PREFIX):
            return False
        if msg.from_mesh:
            return False  # a mesh text: the mesh handler answers over the radio
        if extract_query(msg.text) is None:
            return False
        now = time.time()
        self._seen = {k: v for k, v in self._seen.items() if now - v < 600}
        key = msg.uid or f"{msg.sender_uid}:{msg.text}"
        if key in self._seen:
            return False  # same GeoChat delivered twice
        self._seen[key] = now
        return True

    def on_chat(self, msg: GeoChatMsg):
        if not self.wants(msg):
            return
        logger.info("GeoChat query from %s in %s: %s", msg.sender_callsign,
                    "DM" if msg.direct else msg.chatroom, extract_query(msg.text))
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            logger.warning("Too many pending GeoChat queries, dropping one from %s", msg.sender_callsign)
            self.reply(msg, "ADVISOR busy, try again shortly")

    def answer(self, msg: GeoChatMsg) -> str:
        query = extract_query(msg.text) or ""
        state = self._listener.get_state()
        try:
            context = build_context(state)
            prompt = f"{context}\n\nQuery from {msg.sender_callsign}: {query}{CHAT_STYLE}"
            text = chat_text(self._engine.generate(SYSTEM_PROMPT, prompt, max_tokens=512))
        except Exception:
            logger.exception("Advisor failed to answer GeoChat %r", query)
            text = ""
        if not text or looks_like_error(text):
            if text:
                logger.warning("Not sending setup/error text to chat: %s", text)
            return build_sitrep(state)
        if not str(getattr(self._engine, "last_backend", "") or "").startswith("template"):
            text = ensure_categories(text, state)
        return text

    def reply(self, msg: GeoChatMsg, text: str):
        lat, lon = self._position()
        parts = split_chat(text, reply_label(extract_query(msg.text) or ""))
        for i, part in enumerate(parts):
            if i:
                time.sleep(self.part_gap_s)  # keep them in order in ATAK
            self._send(build_reply(self.own_uid, self.own_callsign, msg, part, lat, lon))
            self.sent += 1
        self.replies += 1
        logger.info("GeoChat reply to %s (%s, %d chars in %d message%s)", msg.sender_callsign,
                    "DM" if msg.direct else msg.chatroom, len(text), len(parts), "" if len(parts) == 1 else "s")

    def handle(self, msg: GeoChatMsg) -> bool:
        """Answer one queued query unless the mesh already has it. Returns True if replied."""
        wait = msg.received + self._hold_s - time.time()
        if wait > 0:
            time.sleep(wait)
        query = extract_query(msg.text) or ""
        if not self._recent.claim("geochat", query, msg.sender_callsign):
            logger.info("GeoChat query %r also came over the mesh: answered there", query)
            return False
        self.reply(msg, self.answer(msg))
        return True

    def _worker(self):
        while self._running:
            try:
                msg = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self.handle(msg)
            except Exception:
                logger.exception("Failed to answer GeoChat query from %s", msg.sender_callsign)
