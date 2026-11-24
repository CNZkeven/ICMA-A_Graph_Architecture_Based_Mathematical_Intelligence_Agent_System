"""Record real ICMA solves for the curated "try a competition problem" cards.

The staged home-page examples replay these recordings with demo pacing, so the
stage payloads shown on screen (classification, retrieved references,
reasoning steps, Python code and stdout) are what the real pipeline produced.

The API key is read from the environment and never written to disk:

    INTERN_API_KEY=... python -m demo_server.record_staged \
        --spec spec.json --out-dir demo_server/staged

``spec.json`` is a JSON list of ``{"id": ..., "problem": ...}`` objects.  Run
from the repository root: the retrieval node opens ``database/`` relative to
the working directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class RecordingSink:
    """In-memory stand-in for ``EventSink`` that keeps each event's offset."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start = time.monotonic()
        self.events: list[dict] = []

    def emit(self, event_type, node=None, payload=None, round_number=0, attempt=None):
        with self._lock:
            event = {
                "t": round(time.monotonic() - self._start, 3),
                "type": event_type,
                "node": node,
                "payload": payload or {},
            }
            self.events.append(event)
        return event


def record(example_id: str, problem: str, key: str) -> dict:
    from demo_server.staged import ROOT_TOKEN, rebase_paths
    from llm_client import InternChatClient
    from user_agent import ReasoningAgent

    # Same client settings as the demo's live path (run_service._live).
    client = InternChatClient(api_key=key, timeout=280, retry=1)
    agent = ReasoningAgent(client)
    sink = RecordingSink()
    started = time.monotonic()
    result = agent.solve(problem, {"idx": -1, "source": "web_demo"}, event_sink=sink)
    recording = {
        "id": example_id,
        "problem": problem,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": client.model,
        "wall_s": round(time.monotonic() - started, 2),
        "events": sink.events,
        "result": result,
    }
    # Keep this machine's checkout path out of the committed recording.
    return rebase_paths(recording, str(ROOT), ROOT_TOKEN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, help="JSON list of {id, problem}")
    parser.add_argument("--out-dir", required=True, help="Directory for <id>.json recordings")
    args = parser.parse_args()

    key = os.environ.get("INTERN_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set INTERN_API_KEY in the environment.")
    if not (Path("database") / "chroma.sqlite3").exists():
        raise SystemExit("Run from the repository root (database/chroma.sqlite3 not found).")

    specs = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def run(spec: dict) -> None:
        try:
            recording = record(spec["id"], spec["problem"], key)
        except Exception as exc:  # noqa: BLE001 - report and keep the other recordings.
            print(f"[{spec['id']}] failed: {exc}", flush=True)
            return
        text = json.dumps(recording, ensure_ascii=False, indent=2)
        if key in text:
            print(f"[{spec['id']}] not written: recording contains the API key", flush=True)
            return
        (out_dir / f"{spec['id']}.json").write_text(text + "\n", encoding="utf-8")
        answer = recording["result"].get("final_response", "")
        print(f"[{spec['id']}] done in {recording['wall_s']}s: {answer[:160]!r}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, len(specs))) as pool:
        list(pool.map(run, specs))


if __name__ == "__main__":
    main()
