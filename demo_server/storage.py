"""Small SQLite persistence layer for the local ICMA demo.

The demo is intentionally single-machine, but every request opens its own
connection so background solver threads and the SSE reader do not share a
SQLite connection across threads or processes.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("ICMA_DEMO_DB", ROOT / "demo_data" / "icma-demo.sqlite3"))
_INIT_LOCK = threading.Lock()
TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def init_db() -> None:
    with _INIT_LOCK:
        with _connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'live',
                    run_id TEXT,
                    trace_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    request_id TEXT,
                    problem TEXT NOT NULL,
                    resolved_problem TEXT NOT NULL,
                    path TEXT NOT NULL DEFAULT 'solve',
                    mode TEXT NOT NULL,
                    example_id TEXT,
                    status TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_request
                    ON runs(conversation_id, request_id)
                    WHERE request_id IS NOT NULL;
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    seq INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    node TEXT,
                    node_run_id TEXT,
                    round INTEGER NOT NULL DEFAULT 0,
                    attempt INTEGER NOT NULL DEFAULT 1,
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(run_id, seq)
                );
                """
            )


def create_conversation(title: str = "新聊天") -> dict[str, Any]:
    conversation_id = str(uuid.uuid4())
    stamp = now_iso()
    with _connect() as db:
        db.execute(
            "INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)",
            (conversation_id, title, stamp, stamp),
        )
    return {"id": conversation_id, "title": title, "created_at": stamp, "updated_at": stamp}


def list_conversations() -> list[dict[str, Any]]:
    with _connect() as db:
        rows = db.execute(
            "SELECT c.id,c.title,c.created_at,c.updated_at,COUNT(m.id) AS message_count "
            "FROM conversations c LEFT JOIN messages m ON m.conversation_id=c.id "
            "GROUP BY c.id ORDER BY c.updated_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def delete_conversation(conversation_id: str) -> bool:
    """Delete one chat and all of its runs, events, and messages."""
    with _connect() as db:
        cursor = db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        return cursor.rowcount > 0


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    with _connect() as db:
        conversation = db.execute(
            "SELECT id,title,created_at,updated_at FROM conversations WHERE id=?",
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            return None
        messages = db.execute(
            "SELECT m.id,m.role,m.content,m.mode,m.run_id,m.trace_json,m.created_at,"
            "r.status AS run_status FROM messages m LEFT JOIN runs r ON r.id=m.run_id "
            "WHERE m.conversation_id=? ORDER BY m.created_at, m.rowid",
            (conversation_id,),
        ).fetchall()
    data = dict(conversation)
    data["messages"] = []
    for row in messages:
        item = dict(row)
        try:
            item["trace"] = json.loads(item.pop("trace_json") or "[]")
        except json.JSONDecodeError:
            item["trace"] = []
            item.pop("trace_json", None)
        data["messages"].append(item)
    return data


def update_conversation(conversation_id: str, title: str | None = None) -> None:
    with _connect() as db:
        if title is None:
            db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now_iso(), conversation_id))
        else:
            db.execute(
                "UPDATE conversations SET title=?,updated_at=? WHERE id=?",
                (title, now_iso(), conversation_id),
            )


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    mode: str = "live",
    run_id: str | None = None,
    trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message_id = str(uuid.uuid4())
    stamp = now_iso()
    with _connect() as db:
        db.execute(
            "INSERT INTO messages(id,conversation_id,role,content,mode,run_id,trace_json,created_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (message_id, conversation_id, role, content, mode, run_id, json.dumps(trace or [], ensure_ascii=False), stamp),
        )
        db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (stamp, conversation_id))
    return {
        "id": message_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "mode": mode,
        "run_id": run_id,
        "trace": trace or [],
        "created_at": stamp,
    }


def create_run(
    conversation_id: str,
    problem: str,
    resolved_problem: str,
    mode: str,
    path: str = "solve",
    example_id: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    with _connect() as db:
        db.execute(
            "INSERT INTO runs(id,conversation_id,request_id,problem,resolved_problem,path,mode,example_id,status) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (run_id, conversation_id, request_id, problem, resolved_problem, path, mode, example_id, "queued"),
        )
    return get_run(run_id) or {"id": run_id, "status": "queued"}


def find_run_by_request(conversation_id: str, request_id: str | None) -> dict[str, Any] | None:
    if not request_id:
        return None
    with _connect() as db:
        row = db.execute(
            "SELECT * FROM runs WHERE conversation_id=? AND request_id=?",
            (conversation_id, request_id),
        ).fetchone()
    return _run_row(row) if row else None


def update_run(run_id: str, **fields: Any) -> None:
    allowed = {
        "status", "started_at", "completed_at", "result_json", "error", "resolved_problem", "path"
    }
    values = [(key, value) for key, value in fields.items() if key in allowed]
    if not values:
        return
    assignments = ",".join(f"{key}=?" for key, _ in values)
    with _connect() as db:
        db.execute(
            f"UPDATE runs SET {assignments} WHERE id=?",
            [value for _, value in values] + [run_id],
        )


def get_run(run_id: str) -> dict[str, Any] | None:
    with _connect() as db:
        row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    return _run_row(row) if row else None


def finish_run(
    run_id: str,
    status: str,
    result: dict[str, Any],
    *,
    error: str | None = None,
    save_message: bool = True,
) -> dict[str, Any] | None:
    """Commit the answer, terminal event and status together, exactly once.

    A worker may be killed during finalization. SQLite then either rolls back
    the entire transaction or commits a complete result, so cancellation can
    never create a duplicate assistant message or overwrite a finished run.
    """
    if status not in TERMINAL_RUN_STATUSES:
        raise ValueError(f"not a terminal run status: {status}")
    stamp = now_iso()
    with _connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            return None
        if row["status"] in TERMINAL_RUN_STATUSES:
            return _run_row(row)
        if save_message:
            db.execute(
                "INSERT INTO messages(id,conversation_id,role,content,mode,run_id,trace_json,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), row["conversation_id"], "assistant", result.get("final_response", ""),
                 row["mode"], run_id, json.dumps(result.get("trace") or [], ensure_ascii=False), stamp),
            )
            db.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (stamp, row["conversation_id"]),
            )
        sequence = db.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE run_id=?", (run_id,),
        ).fetchone()[0]
        summary = {"completed": "运行完成", "failed": error or "运行失败", "cancelled": "用户停止了运行"}[status]
        db.execute(
            "INSERT INTO events(run_id,seq,type,round,attempt,timestamp,payload_json) "
            "VALUES(?,?,?,0,1,?,?)",
            (run_id, sequence, f"run.{status}", stamp, json.dumps({"summary": summary}, ensure_ascii=False)),
        )
        db.execute(
            "UPDATE runs SET status=?,completed_at=?,result_json=?,error=? WHERE id=?",
            (status, stamp, json.dumps(result, ensure_ascii=False), error, run_id),
        )
        final = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    return _run_row(final)


def _run_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    try:
        data["result"] = json.loads(data.pop("result_json") or "{}")
    except json.JSONDecodeError:
        data["result"] = {}
        data.pop("result_json", None)
    return data


def append_event(
    run_id: str,
    seq: int,
    event_type: str,
    node: str | None,
    node_run_id: str | None,
    round_number: int,
    attempt: int,
    timestamp: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    with _connect() as db:
        db.execute(
            "INSERT OR IGNORE INTO events(run_id,seq,type,node,node_run_id,round,attempt,timestamp,payload_json) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (run_id, seq, event_type, node, node_run_id, round_number, attempt, timestamp,
             json.dumps(payload, ensure_ascii=False)),
        )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "seq": seq,
        "type": event_type,
        "node": node,
        "node_run_id": node_run_id,
        "round": round_number,
        "attempt": attempt,
        "timestamp": timestamp,
        "payload": payload,
    }


def get_events(run_id: str, after: int = 0, *, limit: int | None = None) -> list[dict[str, Any]]:
    query = (
        "SELECT seq,type,node,node_run_id,round,attempt,timestamp,payload_json "
        "FROM events WHERE run_id=? AND seq>? ORDER BY seq"
    )
    parameters: list[Any] = [run_id, after]
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(limit)
    with _connect() as db:
        rows = db.execute(query, parameters).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        try:
            item["payload"] = json.loads(item.pop("payload_json") or "{}")
        except json.JSONDecodeError:
            item["payload"] = {}
            item.pop("payload_json", None)
        item["schema_version"] = 1
        item["run_id"] = run_id
        item["type"] = item.pop("type")
        result.append(item)
    return result
