"""Recent queries and answers (web, ATAK chat, mesh), newest first, for the web page.

In memory only: the page shows the latest SITREP and what people asked in chat; nothing is
written to disk.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field

MAX_ENTRIES = 50


@dataclass
class Entry:
    source: str  # "web" | "chat" | "mesh"
    who: str
    query: str
    answer: str
    backend: str = ""
    ts: float = field(default_factory=time.time)


class History:
    def __init__(self, maxlen: int = MAX_ENTRIES):
        self._items: deque[Entry] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, source: str, who: str, query: str, answer: str, backend: str | None = "") -> Entry:
        e = Entry(source=source, who=who or "", query=query or "", answer=answer or "", backend=backend or "")
        with self._lock:
            self._items.appendleft(e)
        return e

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return [asdict(e) for e in list(self._items)[:limit]]

    def clear(self):
        with self._lock:
            self._items.clear()


HISTORY = History()
