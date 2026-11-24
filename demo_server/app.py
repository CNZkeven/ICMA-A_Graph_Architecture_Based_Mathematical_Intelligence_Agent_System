"""FastAPI application for the ICMA local web demo."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

# Support both ``python -m demo_server.app`` and the commonly used
# ``python app.py`` from inside the demo_server directory.  Relative imports
# need the package's parent on sys.path when this file is run as a script.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from demo_server import storage
    from demo_server.fixtures import COMPETITION_EXAMPLES, EXAMPLES
    from demo_server.run_service import cancel_run, start_run, stop_all_runs
else:
    from . import storage
    from .fixtures import COMPETITION_EXAMPLES, EXAMPLES
    from .run_service import cancel_run, start_run, stop_all_runs


storage.init_db()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
        await asyncio.to_thread(stop_all_runs)


app = FastAPI(title="ICMA Demo API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    mode: str = Field(default="live", pattern="^(live|chat|replay)$")
    example_id: str | None = None
    request_id: str | None = None


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": "icma-demo", "environment": "competition"}


@app.get("/api/config")
def config() -> dict:
    key = os.environ.get("INTERN_API_KEY", "")
    return {
        "key_configured": bool(key),
        "masked_key": _mask_key(key),
        "model": os.environ.get("INTERN_MODEL", "intern-s2"),
    }


@app.get("/api/examples")
def examples() -> list[dict]:
    samples = [
        {**{key: value for key, value in example.items() if key not in {"answer", "summary"}}, "group": "sample"}
        for example in EXAMPLES
    ]
    # Allow-list: aliases, answers and recordings never reach the browser.
    competition = [
        {key: example[key] for key in ("id", "difficulty", "title", "problem", "source")} | {"group": "competition"}
        for example in COMPETITION_EXAMPLES
    ]
    return samples + competition


@app.post("/api/conversations")
def create_conversation() -> dict:
    return storage.create_conversation()


@app.get("/api/conversations")
def conversations() -> list[dict]:
    return storage.list_conversations()


@app.get("/api/conversations/{conversation_id}")
def conversation(conversation_id: str) -> dict:
    result = storage.get_conversation(conversation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return result


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict:
    if not storage.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"deleted": True, "id": conversation_id}


@app.post("/api/conversations/{conversation_id}/runs")
def create_run(
    conversation_id: str,
    body: RunRequest,
    x_icma_key: str | None = Header(default=None),
) -> dict:
    if storage.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    if body.mode == "replay" and not body.example_id:
        raise HTTPException(status_code=400, detail="example_id is required for replay")
    run = start_run(
        conversation_id,
        body.message,
        mode=body.mode,
        example_id=body.example_id,
        request_id=body.request_id,
        api_key=x_icma_key,
    )
    return run


@app.get("/api/runs/{run_id}")
def run(run_id: str) -> dict:
    result = storage.get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="run not found")
    return result


@app.post("/api/runs/{run_id}/cancel")
def cancel(run_id: str) -> dict:
    try:
        result = cancel_run(run_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="run not found")
    return result


@app.get("/api/runs/{run_id}/event-batch")
def event_batch(
    run_id: str,
    response: Response,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=500),
) -> dict:
    """Read incremental events through proxies that cannot carry SSE."""
    current = storage.get_run(run_id)
    if current is None:
        raise HTTPException(status_code=404, detail="run not found")
    batch = storage.get_events(run_id, after, limit=limit + 1)
    response.headers["Cache-Control"] = "no-store"
    return {"events": batch[:limit], "has_more": len(batch) > limit, "status": current["status"]}


@app.get("/api/runs/{run_id}/events")
async def events(run_id: str, request: Request, after: int = 0) -> StreamingResponse:
    if storage.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")

    async def stream():
        cursor = after
        heartbeat = 0
        while True:
            if await request.is_disconnected():
                break
            new_events = storage.get_events(run_id, cursor)
            for event in new_events:
                cursor = max(cursor, int(event["seq"]))
                yield f"id: {cursor}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            current = storage.get_run(run_id)
            if current and current["status"] in {"completed", "failed", "cancelled"} and not new_events:
                break
            heartbeat += 1
            if heartbeat >= 20:
                heartbeat = 0
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def _mask_key(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 10:
        return "•" * len(value)
    return f"{value[:5]}…{value[-4:]}"


WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


@app.get("/")
def index():
    index_file = WEB_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "ICMA demo API is running", "frontend": "npm run dev in web/"}


@app.get("/{asset_path:path}")
def static_asset(asset_path: str):
    candidate = (WEB_DIST / asset_path).resolve()
    if WEB_DIST.exists() and candidate.is_file() and WEB_DIST.resolve() in candidate.parents:
        return FileResponse(candidate)
    raise HTTPException(status_code=404, detail="asset not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
