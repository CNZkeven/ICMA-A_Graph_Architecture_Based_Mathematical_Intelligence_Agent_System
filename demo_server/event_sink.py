"""Thread-safe event adapter shared by replay and real solver runs."""

from __future__ import annotations

import threading
import uuid
from typing import Any

from . import storage


class EventSink:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self._lock = threading.Lock()
        self._seq = 0
        self._attempts: dict[str, int] = {}

    def emit(
        self,
        event_type: str,
        node: str | None = None,
        payload: dict[str, Any] | None = None,
        round_number: int = 0,
        attempt: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            if node and attempt is None:
                attempt = self._attempts.get(node, 0) + (1 if event_type == "node.started" else 0)
                self._attempts[node] = attempt
            event = storage.append_event(
                self.run_id,
                self._seq,
                event_type,
                node,
                str(uuid.uuid4()) if node else None,
                round_number,
                attempt or 1,
                storage.now_iso(),
                payload or {},
            )
        return event
