"""Run orchestration for streaming chat, replay and real ICMA solving."""

from __future__ import annotations

import multiprocessing
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from . import storage
from .conversation_service import resolve_request
from .event_sink import EventSink
from .fixtures import get_example, replay_events
from .staged import match_staged_example, run_staged


TERMINAL = storage.TERMINAL_RUN_STATUSES
RUN_EXECUTOR = "process"
_WORKERS: dict[str, Any] = {}
_WORKERS_LOCK = threading.Lock()

CHAT_SYSTEM_PROMPT = """你是 ICMA 数学智能体，面向数学学习、问题求解和证明，提供严谨、清晰、可核查的帮助。
你具备代数、几何、微积分、概率统计、数论等领域的数学推理能力。介绍自己时明确自己的 ICMA 数学智能体身份，不虚构开发机构、模型来源或工具执行经历。
当前处于直接对话模式：由你独立思考并回答，模型自身的思考模式已开启；本轮没有运行完整 ICMA 协作框架，也没有实际调用题库检索、Python 执行或交叉验证工具。可以说明验证方法，但只有获得真实工具结果时才能声称已经执行验证。
结合聊天历史理解用户的追问、条件修改和指定解法；说明变量、定义域、假设及单位。题目信息不足时先指出缺失条件，不擅自补充。
解题时给出清楚的思路、关键公式、必要推导或证明，以及明确的最终结论。检查适用条件、边界和特殊情形，区分精确值与近似值；不能确定的结论应如实说明。
用户要求解释某一步时，针对具体疑问解释依据；发现前文错误时明确纠正。用户只需要简答时避免冗长展开。
使用用户的语言交流，默认中文；采用 Markdown，行内数学公式用 $...$，独立公式用 $$...$$。"""


def _conversation_title(message: str) -> str:
    """Title a LaTeX problem by the prose before its first formula.

    Mirrors ``conversationTitle`` in web/src/main.jsx so the optimistic title
    and the stored one agree.
    """
    text = " ".join(message.split())
    lead = text.split("$", 1)[0].rstrip(" ，。：:；;、,.")
    return (lead if "$" in text and lead else text)[:28].rstrip()


def start_run(
    conversation_id: str,
    message: str,
    mode: str = "live",
    example_id: str | None = None,
    request_id: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    existing = storage.find_run_by_request(conversation_id, request_id)
    if existing:
        return existing
    # The two competition problems always keep their recorded demonstration,
    # including when they are submitted with the full-agent switch disabled.
    staged = match_staged_example(message) if mode in {"live", "chat"} else None
    if staged:
        mode = "live"
    route = (
        {"path": "chat", "resolved_problem": message, "source_run_id": None}
        if mode == "chat" else resolve_request(conversation_id, message)
    )
    example = get_example(example_id) if mode == "replay" else None
    if example:
        route["resolved_problem"] = example["plain_problem"]
    # A competition card, or the same problem pasted into the composer, plays
    # its recorded real run with staged pacing; the run stays mode="live".
    run = storage.create_run(
        conversation_id,
        message,
        route["resolved_problem"],
        mode,
        path=route["path"],
        example_id=staged["id"] if staged else example_id,
        request_id=request_id,
    )
    storage.add_message(conversation_id, "user", message, mode=mode, run_id=run["id"])
    current_conversation = storage.get_conversation(conversation_id) or {}
    if current_conversation.get("title") == "新聊天":
        title = _conversation_title(message)
        if title:
            storage.update_conversation(conversation_id, title=title)
    args = (run["id"], conversation_id, route, mode, example_id, api_key, staged)
    if RUN_EXECUTOR == "thread":
        worker = threading.Thread(
            target=_execute,
            args=args,
            daemon=True,
            name=f"icma-run-{run['id'][:8]}",
        )
    else:
        worker = multiprocessing.get_context("spawn").Process(
            target=_execute_in_process,
            args=(str(storage.DB_PATH), *args),
            name=f"icma-run-{run['id'][:8]}",
        )
    try:
        with _WORKERS_LOCK:
            _WORKERS[run["id"]] = worker
            worker.start()
    except Exception as exc:  # noqa: BLE001 - report worker startup failures in run state.
        with _WORKERS_LOCK:
            _WORKERS.pop(run["id"], None)
        storage.finish_run(run["id"], "failed", {}, error=str(exc), save_message=False)
        return storage.get_run(run["id"]) or run
    threading.Thread(
        target=_reap_worker,
        args=(run["id"], worker),
        daemon=True,
        name=f"icma-reaper-{run['id'][:8]}",
    ).start()
    return run


def _execute_in_process(db_path: str, *args: Any) -> None:
    """Initialize child-process storage explicitly, including test DB paths."""
    if os.name != "nt":
        os.setsid()
    storage.DB_PATH = Path(db_path)
    storage.init_db()
    _execute(*args)


def _reap_worker(run_id: str, worker: Any) -> None:
    worker.join()
    with _WORKERS_LOCK:
        if _WORKERS.get(run_id) is worker:
            _WORKERS.pop(run_id, None)
            if not isinstance(worker, threading.Thread):
                run = storage.get_run(run_id)
                if run is not None and run["status"] not in TERMINAL:
                    storage.finish_run(
                        run_id, "failed", {}, error="运行进程意外退出，请重试。", save_message=False,
                    )


def stop_all_runs() -> None:
    """Close live processes when the API server shuts down."""
    with _WORKERS_LOCK:
        active = [run_id for run_id, worker in _WORKERS.items() if not isinstance(worker, threading.Thread)]
    for run_id in active:
        cancel_run(run_id)
        # A graph can finish while a timed-out model/tool thread is still
        # unwinding. Such a worker is terminal in storage but still needs to
        # exit before multiprocessing will let the API process shut down.
        with _WORKERS_LOCK:
            worker = _WORKERS.get(run_id)
            if worker is not None:
                if worker.is_alive():
                    _terminate_worker(worker)
                    worker.join(timeout=5)
                if not worker.is_alive():
                    _WORKERS.pop(run_id, None)


def cancel_run(run_id: str) -> dict[str, Any] | None:
    """Terminate one active worker and persist everything it emitted so far."""
    with _WORKERS_LOCK:
        run = storage.get_run(run_id)
        if run is None or run["status"] in TERMINAL:
            return run
        worker = _WORKERS.get(run_id)
        if isinstance(worker, threading.Thread):
            raise RuntimeError("当前运行器不支持强制终止")
        if worker is not None:
            if worker.is_alive():
                _terminate_worker(worker)
                worker.join(timeout=5)
                if worker.is_alive() and hasattr(worker, "kill"):
                    worker.kill()
                    worker.join(timeout=2)
                if worker.is_alive():
                    raise RuntimeError("运行进程未能及时终止，请稍后重试")
            else:
                worker.join()
            _WORKERS.pop(run_id, None)
        return _persist_cancelled_run(run_id)


def _terminate_worker(worker: Any) -> None:
    # ICMA's Python executor can launch children of its own. Stop the whole
    # tree so generated code does not keep running after the chat is stopped.
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(worker.pid), "/T", "/F"],
                capture_output=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(worker.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # The worker may not have created its session yet.
    if worker.is_alive():
        worker.terminate()


def _persist_cancelled_run(run_id: str) -> dict[str, Any] | None:
    run = storage.get_run(run_id)
    if run is None or run["status"] in TERMINAL:
        return run

    events = storage.get_events(run_id)
    answer = "".join(
        event["payload"].get("content", "")
        for event in events
        if event["type"] == "message.delta"
    )
    reasoning = "".join(
        event["payload"].get("content", "")
        for event in events
        if event["type"] == "message.reasoning.delta"
    )
    trace = _trace_from_events(events)
    trace.extend(_model_reasoning_trace(reasoning, status="cancelled"))
    for item in trace:
        if item.get("status") == "running":
            item["status"] = "cancelled"
            item["content"] = item.get("content") or "运行在此阶段停止"

    answer = answer or "生成已停止。"
    return storage.finish_run(
        run_id, "cancelled", {"final_response": answer, "trace": trace, "cancelled": True},
    )


def _trace_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild stage progress in the same shape used by the browser."""
    trace: list[dict[str, Any]] = []
    for event in events:
        if not event.get("node") or not event.get("type", "").startswith("node."):
            continue
        payload = event.get("payload") or {}
        event_type = event["type"]
        status = {
            "node.started": "running",
            "node.degraded": "degraded",
            "node.failed": "failed",
        }.get(event_type, "completed")
        item = {
            "step": event["node"],
            "status": status,
            "content": payload.get("summary", ""),
            "thinking": payload.get("thinking") or {},
            "tools": payload.get("tools") or [],
        }
        if payload.get("elapsed_s") is not None:
            item["elapsed_s"] = payload["elapsed_s"]
        if status == "running" and event.get("timestamp"):
            item["started_at"] = event["timestamp"]
        existing = next(
            (index for index in range(len(trace) - 1, -1, -1)
             if trace[index]["step"] == event["node"] and trace[index]["status"] == "running"),
            None,
        )
        if existing is None:
            trace.append(item)
        else:
            trace[existing] = {**trace[existing], **item}
    return trace


def _execute(
    run_id: str,
    conversation_id: str,
    route: dict[str, Any],
    mode: str,
    example_id: str | None,
    api_key: str | None,
    staged: dict[str, Any] | None = None,
) -> None:
    sink = EventSink(run_id)
    storage.update_run(run_id, status="running", started_at=storage.now_iso())
    sink.emit("run.started", payload={"mode": mode, "path": route["path"]})
    try:
        if staged is not None:
            result = run_staged(staged, sink)
        elif mode == "replay":
            result = _replay(example_id, sink)
        elif mode == "chat":
            result = _chat(conversation_id, api_key, sink)
        else:
            result = _live(route["resolved_problem"], api_key, sink)
        storage.finish_run(run_id, "completed", result)
    except Exception as exc:  # noqa: BLE001 - surface readable demo failure.
        events = storage.get_events(run_id)
        partial = "".join(
            event["payload"].get("content", "")
            for event in events
            if event["type"] == "message.delta"
        )
        reasoning = "".join(
            event["payload"].get("content", "")
            for event in events
            if event["type"] == "message.reasoning.delta"
        )
        trace = _model_reasoning_trace(reasoning, status="failed")
        answer = partial or "本次回答未完成，请重试。"
        storage.finish_run(
            run_id, "failed", {"final_response": answer, "trace": trace},
            error=str(exc), save_message=bool(partial or reasoning),
        )


def _replay(example_id: str | None, sink: EventSink) -> dict[str, Any]:
    example = get_example(example_id) or get_example("integral")
    assert example is not None
    trace = []
    delay = float(os.environ.get("ICMA_REPLAY_DELAY", "0.42"))
    stages = replay_events(example)
    # Emit reasoning and Python as the same round so the graph UI can show the
    # real parallel shape even though the fixture is played on one worker.
    for stage in stages:
        node = stage["node"]
        sink.emit("node.started", node=node, payload={"summary": f"{node} 运行中"})
        time.sleep(delay)
        sink.emit(
            "node.completed",
            node=node,
            payload={
                "summary": stage["summary"],
                "elapsed_s": stage["elapsed_s"],
                "thinking": stage.get("thinking", {}),
                "tools": stage.get("tools", []),
            },
        )
        trace.append({
            "step": node,
            "status": "completed",
            "content": stage["summary"],
            "elapsed_s": stage["elapsed_s"],
            "thinking": stage.get("thinking", {}),
            "tools": stage.get("tools", []),
        })
    return {
        "final_response": example["answer"],
        "trace": trace,
        "metadata": {"replay": True, "check": example["check"], "example_id": example["id"]},
    }


def _client(api_key: str | None, *, retry: int = 1):
    from llm_client import InternChatClient

    key = (api_key or os.environ.get("INTERN_API_KEY", "")).strip()
    if not key:
        raise RuntimeError("未设置 API Key。请在左下角 Key 设置中填写后重试。")
    return InternChatClient(api_key=key, timeout=280, retry=retry)


def _chat(conversation_id: str, api_key: str | None, sink: EventSink) -> dict[str, Any]:
    conversation = storage.get_conversation(conversation_id) or {"messages": []}
    messages = [{
        "role": "system",
        "content": CHAT_SYSTEM_PROMPT,
    }]
    messages.extend(
        {"role": item["role"], "content": item["content"]}
        for item in conversation["messages"]
        if item["role"] in {"user", "assistant"} and item["content"]
    )
    parts = []
    reasoning_parts = []

    def on_reasoning(content: str) -> None:
        reasoning_parts.append(content)
        sink.emit("message.reasoning.delta", payload={"content": content})

    for content in _client(api_key, retry=2).stream_chat(messages, on_reasoning=on_reasoning):
        parts.append(content)
        sink.emit("message.delta", payload={"content": content})
    answer = "".join(parts)
    if not answer.strip():
        raise RuntimeError("模型未返回回答，请稍后重试。")
    return {
        "final_response": answer,
        "trace": _model_reasoning_trace("".join(reasoning_parts)),
        "metadata": {"replay": False, "chat": True},
    }


def _model_reasoning_trace(content: str, status: str = "completed") -> list[dict[str, Any]]:
    if not content:
        return []
    return [{
        "step": "model_reasoning",
        "status": status,
        "content": {"failed": "模型思考中断", "cancelled": "模型思考已停止"}.get(status, "模型思考完成"),
        "thinking": {"analysis": content},
    }]


def _live(problem: str, api_key: str | None, sink: EventSink) -> dict[str, Any]:
    from user_agent import ReasoningAgent

    agent = ReasoningAgent(_client(api_key))
    result = agent.solve(problem, {"idx": -1, "source": "web_demo"}, event_sink=sink)
    trace = result.get("trace") or []
    if not trace:
        trace = [{"step": "coordinator", "status": "completed", "content": "已返回结果。"}]
    return {"final_response": result.get("final_response", ""), "trace": trace, "metadata": {"replay": False}}
