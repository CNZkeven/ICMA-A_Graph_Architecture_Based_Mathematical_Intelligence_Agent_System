"""Offline contract tests for the local web demo.

Runs use deterministic fixtures or mocked model clients, so the suite never
sends a model request or requires a live API key.
"""

from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from demo_server import run_service, staged, storage
from demo_server.app import app
from demo_server.fixtures import COMPETITION_EXAMPLES


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "demo.sqlite3")
    # These contract tests monkeypatch model and solver clients in-process.
    monkeypatch.setattr(run_service, "RUN_EXECUTOR", "thread")
    storage.init_db()
    with TestClient(app) as test_client:
        yield test_client


def wait_for_run(client: TestClient, run_id: str) -> dict:
    for _ in range(80):
        result = client.get(f"/api/runs/{run_id}").json()
        if result["status"] in {"completed", "failed", "cancelled"}:
            return result
        time.sleep(0.05)
    pytest.fail("replay run did not reach a terminal state")


def test_cancel_terminates_isolated_worker_and_saves_partial_trace(client, monkeypatch):
    monkeypatch.setattr(run_service, "RUN_EXECUTOR", "process")
    monkeypatch.setenv("ICMA_REPLAY_DELAY", "30")
    conversation = client.post("/api/conversations").json()
    run = client.post(
        f"/api/conversations/{conversation['id']}/runs",
        json={
            "message": "计算积分 ∫₀¹ x² dx。",
            "mode": "replay",
            "example_id": "integral",
        },
    ).json()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        events = storage.get_events(run["id"])
        if any(event["type"] == "node.started" for event in events):
            break
        time.sleep(0.05)
    else:
        pytest.fail("isolated worker did not start a replay stage")

    response = client.post(f"/api/runs/{run['id']}/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert client.post(f"/api/runs/{run['id']}/cancel").json()["status"] == "cancelled"

    events = storage.get_events(run["id"])
    assert events[-1]["type"] == "run.cancelled"
    assert events[-1]["seq"] > events[-2]["seq"]
    history = client.get(f"/api/conversations/{conversation['id']}").json()
    assistant = next(message for message in history["messages"] if message["role"] == "assistant")
    assert assistant["content"] == "生成已停止。"
    assert assistant["trace"][0]["step"] == "input"
    assert assistant["trace"][0]["status"] == "cancelled"


def test_api_shutdown_stops_an_active_worker(client, monkeypatch):
    monkeypatch.setattr(run_service, "RUN_EXECUTOR", "process")
    monkeypatch.setenv("ICMA_REPLAY_DELAY", "30")
    with TestClient(app) as api_client:
        conversation = api_client.post("/api/conversations").json()
        run = api_client.post(
            f"/api/conversations/{conversation['id']}/runs",
            json={"message": "计算积分", "mode": "replay", "example_id": "integral"},
        ).json()
        worker = run_service._WORKERS[run["id"]]
    assert not worker.is_alive()
    assert storage.get_run(run["id"])["status"] == "cancelled"
    assert storage.get_events(run["id"])[-1]["type"] == "run.cancelled"


@pytest.mark.parametrize("output", ["silent", "reasoning", "answer"])
def test_cancel_interrupts_blocked_model_stream_and_keeps_output(client, monkeypatch, output):
    """Use a real local socket and spawned worker, including a silent stream."""
    monkeypatch.setattr(run_service, "RUN_EXECUTOR", "process")
    connected = threading.Event()
    release = threading.Event()

    class ModelHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            deltas = []
            if output != "silent":
                deltas.append({"reasoning_content": "先检查定义域。"})
            if output == "answer":
                deltas.append({"content": "目前得到的部分答案。"})
            for delta in deltas:
                event = {"choices": [{"delta": delta, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()
            connected.set()
            release.wait(15)

    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    monkeypatch.setenv("INTERN_API_BASE", f"http://127.0.0.1:{server.server_port}/api/v1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    try:
        conversation = client.post("/api/conversations").json()
        run = client.post(
            f"/api/conversations/{conversation['id']}/runs",
            json={"message": "请分析这个函数", "mode": "chat"},
            headers={"X-ICMA-Key": "local-offline-test"},
        ).json()
        assert connected.wait(8)
        expected_type = "message.delta" if output == "answer" else "message.reasoning.delta"
        if output != "silent":
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if any(event["type"] == expected_type for event in storage.get_events(run["id"])):
                    break
                time.sleep(0.02)
            else:
                pytest.fail("model stream was not persisted before cancellation")
        worker = run_service._WORKERS[run["id"]]
        started = time.monotonic()
        response = client.post(f"/api/runs/{run['id']}/cancel")
        assert time.monotonic() - started < 5
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
        assert not worker.is_alive()
        assert not release.is_set(), "model handler is still blocked when cancellation returns"

        events = storage.get_events(run["id"])
        assert events[-1]["type"] == "run.cancelled"
        assert len({event["seq"] for event in events}) == len(events)
        history = client.get(f"/api/conversations/{conversation['id']}").json()
        assistants = [message for message in history["messages"] if message["role"] == "assistant"]
        assert len(assistants) == 1
        assistant = assistants[0]
        assert assistant["run_status"] == "cancelled"
        assert assistant["content"] == ("目前得到的部分答案。" if output == "answer" else "生成已停止。")
        if output != "silent":
            assert assistant["trace"][0]["status"] == "cancelled"
            assert assistant["trace"][0]["thinking"]["analysis"] == "先检查定义域。"
        assert response.json()["result"]["trace"] == assistant["trace"]
        resumed = client.get(f"/api/runs/{run['id']}/events?after={events[-2]['seq']}").text
        assert "run.cancelled" in resumed

        again = client.post(f"/api/runs/{run['id']}/cancel")
        assert again.json() == response.json()
        assert storage.get_events(run["id"]) == events
        # A stopped conversation remains usable for another turn.
        monkeypatch.setenv("ICMA_REPLAY_DELAY", "0")
        next_run = client.post(
            f"/api/conversations/{conversation['id']}/runs",
            json={"message": "继续验证积分", "mode": "replay", "example_id": "integral"},
        ).json()
        assert wait_for_run(client, next_run["id"])["status"] == "completed"
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


def test_completion_and_cancel_commit_only_one_result(client):
    conversation = storage.create_conversation()
    run = storage.create_run(conversation["id"], "题目", "题目", "chat")
    barrier = threading.Barrier(2)

    def finalize(status):
        barrier.wait(timeout=3)
        return storage.finish_run(run["id"], status, {"final_response": status, "trace": []})

    with ThreadPoolExecutor(max_workers=2) as pool:
        completed = pool.submit(finalize, "completed")
        cancelled = pool.submit(finalize, "cancelled")
        first, second = completed.result(timeout=5), cancelled.result(timeout=5)
    assert first == second
    assistants = storage.get_conversation(conversation["id"])["messages"]
    assert len(assistants) == 1
    assert assistants[0]["content"] == first["status"]
    assert [event["type"] for event in storage.get_events(run["id"])] == [f"run.{first['status']}"]
    assert client.post(f"/api/runs/{run['id']}/cancel").json() == first
    assert client.post("/api/runs/missing/cancel").status_code == 404


def test_health_examples_and_key_mask(client):
    assert client.get("/api/health").json()["environment"] == "competition"
    examples = client.get("/api/examples").json()
    samples = [item for item in examples if item["group"] == "sample"]
    competition = [item for item in examples if item["group"] == "competition"]
    assert [item["id"] for item in samples] == ["integral", "optimization", "number-theory"]
    assert [item["id"] for item in competition] == [item["id"] for item in COMPETITION_EXAMPLES]
    for item in competition:
        assert set(item) == {"id", "difficulty", "title", "problem", "source", "group"}
    assert client.get("/api/config").json()["key_configured"] is False


def test_replay_persists_messages_and_emits_sse(client):
    conversation = client.post("/api/conversations").json()
    example = client.get("/api/examples").json()[0]
    run = client.post(
        f"/api/conversations/{conversation['id']}/runs",
        json={
            "message": example["plain_problem"],
            "mode": "replay",
            "example_id": example["id"],
            "request_id": "offline-test-1",
        },
    ).json()
    result = wait_for_run(client, run["id"])
    assert result["status"] == "completed"
    assert "1}{3}" in result["result"]["final_response"]
    raw = client.get(f"/api/runs/{run['id']}/events").text
    events = [json.loads(block.split("data: ", 1)[1].splitlines()[0]) for block in raw.split("\n\n") if "data: " in block]
    assert events[0]["type"] == "run.started"
    assert events[-1]["type"] == "run.completed"
    assert sum(event["type"] == "node.completed" for event in events) == 7
    resumed = client.get(f"/api/runs/{run['id']}/events?after={events[4]['seq']}").text
    resumed_events = [json.loads(block.split("data: ", 1)[1].splitlines()[0]) for block in resumed.split("\n\n") if "data: " in block]
    assert resumed_events and resumed_events[0]["seq"] > events[4]["seq"]
    history = client.get(f"/api/conversations/{conversation['id']}").json()
    assert [message["role"] for message in history["messages"]] == ["user", "assistant"]
    assistant_trace = history["messages"][1]["trace"]
    assert assistant_trace[0]["thinking"]
    assert assistant_trace[0]["tools"]
    stages = {item["step"]: item for item in assistant_trace}
    assert "skills_pythonscripts" in stages["classifier"]["thinking"]["category_path"]
    assert len(stages["database_retrieval"]["thinking"]["references"]) == 2
    assert stages["reasoning_agent"]["thinking"]["steps"]
    assert "sympy" in stages["python_agent"]["thinking"]["python_code"]
    assert "最终答案: 1/3" in stages["python_agent"]["thinking"]["stdout"]


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_event_batches_drain_terminal_runs_in_order(client, status):
    conversation = storage.create_conversation()
    run = storage.create_run(conversation["id"], "题目", "题目", "chat")
    for sequence in range(1, 6):
        storage.append_event(
            run["id"], sequence, "message.delta", None, None, 0, 1,
            storage.now_iso(), {"content": f"第{sequence}段"},
        )
    storage.finish_run(run["id"], status, {"final_response": "答案", "trace": []})
    path = f"/api/runs/{run['id']}/event-batch"
    received = []
    after = 0
    for _ in range(3):
        response = client.get(path, params={"after": after, "limit": 2})
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        batch = response.json()
        assert batch["status"] == status
        received.extend(batch["events"])
        after = batch["events"][-1]["seq"]
        assert batch["has_more"] == (after < 6)
    assert received == storage.get_events(run["id"])
    assert [event["seq"] for event in received] == list(range(1, 7))
    assert received[-1]["type"] == f"run.{status}"
    assert client.get(path, params={"after": after}).json() == {
        "events": [], "has_more": False, "status": status,
    }
    assert client.get("/api/runs/missing/event-batch").status_code == 404
    for params in ({"after": -1}, {"limit": 0}, {"limit": 501}):
        assert client.get(path, params=params).status_code == 422


def test_event_polling_exposes_reasoning_and_answer_before_completion(client, monkeypatch):
    ready = threading.Event()
    release = threading.Event()

    class PausedClient:
        def stream_chat(self, _messages, *, on_reasoning):
            on_reasoning("先检查条件。")
            yield "已经得到的第一段。"
            ready.set()
            if not release.wait(5):
                raise RuntimeError("test did not release the model")
            yield "第二段。"

    monkeypatch.setattr(run_service, "_client", lambda _key, **_options: PausedClient())
    conversation = client.post("/api/conversations").json()
    run = client.post(
        f"/api/conversations/{conversation['id']}/runs",
        json={"message": "请解题", "mode": "chat"},
    ).json()
    path = f"/api/runs/{run['id']}/event-batch"
    try:
        assert ready.wait(3)
        initial = client.get(path).json()
        assert initial["status"] == "running"
        assert [event["type"] for event in initial["events"]] == [
            "run.started", "message.reasoning.delta", "message.delta",
        ]
        assert initial["events"][1]["payload"]["content"] == "先检查条件。"
        assert initial["events"][2]["payload"]["content"] == "已经得到的第一段。"
        cursor = initial["events"][-1]["seq"]
        assert client.get(path, params={"after": cursor}).json()["events"] == []
    finally:
        release.set()
    assert wait_for_run(client, run["id"])["status"] == "completed"
    remaining = client.get(path, params={"after": cursor}).json()
    assert remaining["status"] == "completed"
    assert [event["type"] for event in remaining["events"]] == ["message.delta", "run.completed"]
    assert remaining["events"][0]["payload"]["content"] == "第二段。"


def test_request_id_is_idempotent(client):
    conversation = client.post("/api/conversations").json()
    payload = {
        "message": "计算积分 ∫₀¹ x² dx。",
        "mode": "replay",
        "example_id": "integral",
        "request_id": "same-request",
    }
    first = client.post(f"/api/conversations/{conversation['id']}/runs", json=payload).json()
    second = client.post(f"/api/conversations/{conversation['id']}/runs", json=payload).json()
    assert second["id"] == first["id"]
    assert wait_for_run(client, first["id"])["status"] == "completed"


def test_conversation_can_be_deleted(client):
    conversation = client.post("/api/conversations").json()
    assert client.get("/api/conversations").json()[0]["message_count"] == 0
    deleted = client.delete(f"/api/conversations/{conversation['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert client.get(f"/api/conversations/{conversation['id']}").status_code == 404
    assert client.delete(f"/api/conversations/{conversation['id']}").status_code == 404


# ---- staged competition examples ------------------------------------------------

def _event(t, kind, node, **payload):
    return {"t": t, "type": kind, "node": node, "payload": payload}


# Shape of a real recording: sequential stages around one parallel agent round.
SYNTHETIC_RECORDING = {
    "events": [
        _event(0.0, "node.started", "input"),
        _event(0.1, "node.completed", "input", summary="input 完成"),
        _event(0.1, "node.started", "classifier"),
        _event(4.0, "node.completed", "classifier", summary="classifier 完成", thinking={"category": "数学分析"}),
        _event(4.0, "node.started", "database_retrieval"),
        _event(9.0, "node.completed", "database_retrieval", summary="database_retrieval 完成"),
        _event(9.0, "node.started", "reasoning_agent"),
        _event(9.1, "node.started", "python_agent"),
        _event(80.0, "node.completed", "python_agent", summary="python_agent 完成", tools=[{"name": "python_executor"}]),
        _event(160.0, "node.completed", "reasoning_agent", summary="reasoning_agent 完成", thinking={"steps": ["a", "b"]}),
        _event(160.0, "node.started", "cross_validator"),
        _event(160.2, "node.completed", "cross_validator", summary="cross_validator 完成"),
        _event(160.2, "node.started", "coordinator"),
        _event(170.0, "node.completed", "coordinator", summary="coordinator 完成"),
    ],
    "result": {"final_response": "recorded answer", "trace": []},
}


class ListSink:
    def __init__(self):
        self.events = []

    def emit(self, event_type, node=None, payload=None, **_):
        self.events.append({"type": event_type, "node": node, "payload": payload or {}})


def test_staged_problem_matching_ignores_presentation_only():
    card = COMPETITION_EXAMPLES[1]
    typed = "已知可导函数f(x)满足 $f(x)\\cos x + 2\\int_0^x f(t)\\sin t\\,dt = x+1$, 求f(x)"
    assert staged.match_staged_example(card["problem"]) is card
    assert staged.match_staged_example(typed) is card
    assert staged.match_staged_example(card["aliases"][0] + "  ") is card
    assert staged.match_staged_example(card["problem"].replace("x+1$", "x+2$")) is None
    assert staged.match_staged_example("计算积分 ∫₀¹ x² dx。") is None


def test_staged_plan_runs_both_agents_as_one_phase():
    phases = staged.plan_phases(SYNTHETIC_RECORDING["events"])
    assert [[run["node"] for run in phase] for phase in phases] == [
        ["input"], ["classifier"], ["database_retrieval"],
        ["reasoning_agent", "python_agent"], ["cross_validator"], ["coordinator"],
    ]


def test_run_staged_paces_agents_and_stages(monkeypatch):
    monkeypatch.setattr(staged, "load_recording", lambda _id: SYNTHETIC_RECORDING)
    sleeps = []
    sink = ListSink()
    result = staged.run_staged({"id": "synthetic"}, sink, rng=random.Random(7), sleep=sleeps.append)

    durations = {item["step"]: item["elapsed_s"] for item in result["trace"]}
    for node, seconds in durations.items():
        low, high = staged.AGENT_SECONDS if node in staged.AGENT_NODES else staged.STAGE_SECONDS
        assert low <= seconds <= high, node
    # The real sleeps add up to the reported durations, agents overlapping.
    agent_phase = max(durations["reasoning_agent"], durations["python_agent"])
    others = sum(value for key, value in durations.items() if key not in staged.AGENT_NODES)
    assert sum(sleeps) == pytest.approx(agent_phase + others)
    # Both agents start before either finishes.
    kinds = [(event["type"], event["node"]) for event in sink.events]
    agent_events = [kind for kind in kinds if kind[1] in staged.AGENT_NODES]
    assert [kind for kind, _ in agent_events[:2]] == ["node.started", "node.started"]
    # Recorded stage payloads are replayed, with the paced elapsed time.
    reasoning = next(item for item in result["trace"] if item["step"] == "reasoning_agent")
    assert reasoning["thinking"] == {"steps": ["a", "b"]}
    assert reasoning["content"] == "完成 2 步推导，形成候选结论"
    assert result["final_response"] == "recorded answer"


@pytest.mark.parametrize("mode", ["live", "chat"])
def test_competition_card_runs_staged_as_live(client, monkeypatch, mode):
    monkeypatch.setattr(staged, "load_recording", lambda _id: SYNTHETIC_RECORDING)
    monkeypatch.setenv("ICMA_STAGED_TIME_SCALE", "0")
    card = COMPETITION_EXAMPLES[0]
    conversation = client.post("/api/conversations").json()
    run = client.post(
        f"/api/conversations/{conversation['id']}/runs",
        json={"message": card["problem"], "mode": mode, "request_id": "staged-1"},
    ).json()
    result = wait_for_run(client, run["id"])
    assert result["status"] == "completed"
    assert result["mode"] == "live"
    assert result["result"]["final_response"] == "recorded answer"
    history = client.get(f"/api/conversations/{conversation['id']}").json()
    assert history["title"] == "求极限"
    assistant = history["messages"][1]
    assert assistant["mode"] == "live"
    assert [item["step"] for item in assistant["trace"]][:3] == ["input", "classifier", "database_retrieval"]


@pytest.mark.parametrize("example", COMPETITION_EXAMPLES, ids=lambda example: example["id"])
def test_shipped_recordings_are_usable(example):
    recording = staged.load_recording(example["id"])
    assert recording["result"]["final_response"].strip()
    phases = [{run["node"] for run in phase} for phase in staged.plan_phases(recording["events"])]
    assert {"reasoning_agent", "python_agent"} in phases
    assert sum(phase == {"reasoning_agent", "python_agent"} for phase in phases) == 1


def test_chat_streams_before_completion_and_uses_conversation_history(client, monkeypatch):
    import llm_client

    conversation = client.post("/api/conversations").json()
    conversation_id = conversation["id"]
    storage.add_message(conversation_id, "user", "求 f(x) 的导数。")
    storage.add_message(conversation_id, "assistant", "$f'(x)=2x$。", trace=[{"step": "reasoning_agent"}])
    ready = threading.Event()
    release = threading.Event()
    captured = {}

    class StreamingClient:
        def __init__(self, **kwargs):
            captured["key"] = kwargs["api_key"]
            captured["retry"] = kwargs["retry"]

        def stream_chat(self, messages, **_options):
            captured["messages"] = messages
            yield "结合上一轮，"
            ready.set()
            if not release.wait(5):
                raise RuntimeError("test stream was not released")
            yield "导数为 $2x$。"

    monkeypatch.setattr(llm_client, "InternChatClient", StreamingClient)
    payload = {"message": "解释这个结果", "mode": "chat", "request_id": "chat-stream"}
    try:
        run = client.post(
            f"/api/conversations/{conversation_id}/runs",
            json=payload,
            headers={"X-ICMA-Key": "test-key"},
        ).json()
        assert ready.wait(3)
        assert client.get(f"/api/runs/{run['id']}").json()["status"] == "running"
        deltas = [item for item in storage.get_events(run["id"]) if item["type"] == "message.delta"]
        assert [item["payload"]["content"] for item in deltas] == ["结合上一轮，"]
        duplicate = client.post(f"/api/conversations/{conversation_id}/runs", json=payload).json()
        assert duplicate["id"] == run["id"]
    finally:
        release.set()
    result = wait_for_run(client, run["id"])
    assert result["status"] == "completed"
    assert result["mode"] == "chat"
    assert result["path"] == "chat"
    assert result["result"]["final_response"] == "结合上一轮，导数为 $2x$。"
    assert result["result"]["trace"] == []
    assert captured["key"] == "test-key"
    assert captured["retry"] == 2
    assert captured["messages"][0]["role"] == "system"
    assert "你是 ICMA 数学智能体" in captured["messages"][0]["content"]
    assert "直接对话模式" in captured["messages"][0]["content"]
    assert "没有实际调用题库检索、Python 执行或交叉验证工具" in captured["messages"][0]["content"]
    assert captured["messages"][1:] == [
        {"role": "user", "content": "求 f(x) 的导数。"},
        {"role": "assistant", "content": "$f'(x)=2x$。"},
        {"role": "user", "content": "解释这个结果"},
    ]
    events = storage.get_events(run["id"])
    assert not any(item["type"].startswith("node.") for item in events)
    resumed = client.get(f"/api/runs/{run['id']}/events?after={deltas[0]['seq']}").text
    assert "结合上一轮，" not in resumed
    assert "导数为 $2x$。" in resumed
    history = client.get(f"/api/conversations/{conversation_id}").json()["messages"]
    assert len(history) == 4
    assert history[-1]["mode"] == "chat"
    assert history[-1]["trace"] == []


def test_chat_failure_preserves_partial_answer(client, monkeypatch):
    class InterruptedClient:
        def stream_chat(self, messages, **_options):
            yield "已经生成的回答。"
            raise RuntimeError("测试连接中断")

    monkeypatch.setattr(run_service, "_client", lambda _key, **_options: InterruptedClient())
    conversation = client.post("/api/conversations").json()
    run = client.post(
        f"/api/conversations/{conversation['id']}/runs",
        json={"message": "你好", "mode": "chat"},
    ).json()
    result = wait_for_run(client, run["id"])
    assert result["status"] == "failed"
    assert result["error"] == "测试连接中断"
    assert result["result"]["final_response"] == "已经生成的回答。"
    assert storage.get_events(run["id"])[-1]["type"] == "run.failed"
    history = client.get(f"/api/conversations/{conversation['id']}").json()["messages"]
    assert history[-1]["content"] == "已经生成的回答。"
    assert history[-1]["run_id"] == run["id"]


@pytest.mark.parametrize("has_key", [True, False])
def test_chat_without_answer_fails_cleanly(client, monkeypatch, has_key):
    class EmptyClient:
        def stream_chat(self, messages, **_options):
            return iter(())

    if has_key:
        monkeypatch.setattr(run_service, "_client", lambda _key, **_options: EmptyClient())
    else:
        monkeypatch.delenv("INTERN_API_KEY", raising=False)
    conversation = client.post("/api/conversations").json()
    run = client.post(
        f"/api/conversations/{conversation['id']}/runs",
        json={"message": "你好", "mode": "chat"},
    ).json()
    result = wait_for_run(client, run["id"])
    assert result["status"] == "failed"
    expected_error = "未返回回答" if has_key else "未设置 API Key"
    assert expected_error in result["error"]


@pytest.mark.parametrize("follow_up", [False, True])
def test_enabled_agent_uses_icma_for_new_questions_and_explanations(client, monkeypatch, follow_up):
    captured = []

    def live(problem, api_key, sink):
        captured.append(problem)
        sink.emit("node.completed", node="reasoning_agent", payload={"summary": "真实框架路径"})
        return {"final_response": "ICMA answer", "trace": [{"step": "reasoning_agent"}]}

    monkeypatch.setattr(run_service, "_live", live)
    conversation = client.post("/api/conversations").json()
    if follow_up:
        storage.add_message(conversation["id"], "user", "求导数")
        storage.add_message(conversation["id"], "assistant", "上一轮答案")
    message = "解释第二步为什么这样做" if follow_up else "求 x^3 的导数"
    run = client.post(
        f"/api/conversations/{conversation['id']}/runs",
        json={"message": message},
    ).json()
    result = wait_for_run(client, run["id"])
    assert result["status"] == "completed"
    assert result["mode"] == "live"
    assert message in captured[0]
    if follow_up:
        assert "上一轮答案" in captured[0]
    assert result["result"]["final_response"] == "ICMA answer"


def test_chat_reasoning_is_streamed_before_answer_and_persisted(client, monkeypatch):
    ready = threading.Event()
    release = threading.Event()

    class ThinkingClient:
        def stream_chat(self, messages, *, on_reasoning):
            on_reasoning("先检查条件。")
            ready.set()
            if not release.wait(5):
                raise RuntimeError("test reasoning was not released")
            on_reasoning("再利用 $f'(x)=2x$ 核对结果。")
            yield "结论为 $2x$。"

    monkeypatch.setattr(run_service, "_client", lambda _key, **_options: ThinkingClient())
    conversation = client.post("/api/conversations").json()
    try:
        run = client.post(
            f"/api/conversations/{conversation['id']}/runs",
            json={"message": "求 x^2 的导数", "mode": "chat"},
        ).json()
        assert ready.wait(3)
        assert storage.get_run(run["id"])["status"] == "running"
        initial = storage.get_events(run["id"])
        assert [event["type"] for event in initial] == ["run.started", "message.reasoning.delta"]
        assert initial[-1]["payload"]["content"] == "先检查条件。"
    finally:
        release.set()
    result = wait_for_run(client, run["id"])
    assert result["status"] == "completed"
    trace = result["result"]["trace"]
    assert trace[0]["step"] == "model_reasoning"
    assert trace[0]["status"] == "completed"
    assert trace[0]["thinking"]["analysis"] == "先检查条件。再利用 $f'(x)=2x$ 核对结果。"
    assert result["result"]["final_response"] == "结论为 $2x$。"
    history = storage.get_conversation(conversation["id"])["messages"]
    assert history[-1]["trace"] == trace
    resumed = client.get(f"/api/runs/{run['id']}/events?after={initial[-1]['seq']}").text
    assert "先检查条件。" not in resumed
    assert "再利用" in resumed
    assert "结论为" in resumed
    assert not any(event["type"].startswith("node.") for event in storage.get_events(run["id"]))


@pytest.mark.parametrize("with_answer", [False, True])
def test_failed_chat_keeps_visible_reasoning(client, monkeypatch, with_answer):
    class InterruptedThinkingClient:
        def stream_chat(self, messages, *, on_reasoning):
            on_reasoning("已经检查了定义域。")
            if with_answer:
                yield "部分结论。"
            raise RuntimeError("测试思考中断")

    monkeypatch.setattr(run_service, "_client", lambda _key, **_options: InterruptedThinkingClient())
    conversation = client.post("/api/conversations").json()
    run = client.post(
        f"/api/conversations/{conversation['id']}/runs",
        json={"message": "请分析这个函数", "mode": "chat"},
    ).json()
    result = wait_for_run(client, run["id"])
    assert result["status"] == "failed"
    assert result["error"] == "测试思考中断"
    assistant = storage.get_conversation(conversation["id"])["messages"][-1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == ("部分结论。" if with_answer else "本次回答未完成，请重试。")
    assert assistant["trace"][0]["status"] == "failed"
    assert assistant["trace"][0]["thinking"]["analysis"] == "已经检查了定义域。"
    assert result["result"]["trace"] == assistant["trace"]
