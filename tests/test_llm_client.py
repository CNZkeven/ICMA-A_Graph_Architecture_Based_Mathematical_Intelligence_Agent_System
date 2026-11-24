"""Offline checks for the model's SSE transport and partial-stream failures."""

import json

import pytest
import requests

from llm_client import InternChatClient


def frame(content=None, reason=None, **delta):
    if content is not None:
        delta["content"] = content
    return ("data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": reason}]}, ensure_ascii=False)).encode("utf-8")


class Response:
    def __init__(self, lines):
        self.lines = lines
        self.closed = False
        self.consumed = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def raise_for_status(self):
        pass

    def iter_lines(self, **_):
        for line in self.lines:
            self.consumed += 1
            if isinstance(line, Exception):
                raise line
            yield line


def test_stream_yields_utf8_content_incrementally_and_requests_streaming(monkeypatch):
    response = Response([
        b": keep-alive", b"", frame(reasoning_content="internal"),
        frame("第一段"), frame("，第二段。"), frame(reason="stop"), b"data: [DONE]",
    ])
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr(requests, "post", post)
    client = InternChatClient(api_key="test-key", retry=1)
    reasoning = []
    answer = client.stream_chat([{"role": "user", "content": "你好"}], on_reasoning=reasoning.append)
    assert next(answer) == "第一段"
    assert reasoning == ["internal"]
    assert response.consumed == 4
    assert not response.closed
    assert list(answer) == ["，第二段。"]
    assert response.closed
    payload = json.loads(calls[0]["data"])
    assert calls[0]["stream"] is True
    assert payload["stream"] is True
    assert payload["thinking_mode"] is True
    assert calls[0]["timeout"] == (10, 120)


@pytest.mark.parametrize("failure", [
    requests.ConnectionError("connection lost"),
    b'data: {"error":{"message":"upstream error"}}',
    b'data: {"object":"error","message":"upstream error"}',
])
def test_stream_never_retries_after_visible_content(monkeypatch, failure):
    response = Response([frame("partial"), failure])
    calls = []

    def post(*args, **kwargs):
        calls.append(True)
        return response

    monkeypatch.setattr(requests, "post", post)
    answer = InternChatClient(api_key="test-key", retry=3).stream_chat([])
    assert next(answer) == "partial"
    with pytest.raises(RuntimeError, match="流式回答失败"):
        next(answer)
    assert len(calls) == 1
    assert response.closed


def test_stream_never_retries_after_visible_reasoning(monkeypatch):
    response = Response([frame(reasoning_content="已检查题目条件。"), requests.ConnectionError("connection lost")])
    calls = []
    reasoning = []

    def post(*args, **kwargs):
        calls.append(True)
        return response

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(RuntimeError, match="流式回答失败"):
        list(InternChatClient(api_key="test-key", retry=2).stream_chat([], on_reasoning=reasoning.append))
    assert reasoning == ["已检查题目条件。"]
    assert len(calls) == 1
    assert response.closed


@pytest.mark.parametrize("ending", [[], [frame(reason="length")]])
def test_incomplete_stream_does_not_report_success(monkeypatch, ending):
    response = Response([frame("partial"), *ending])
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)
    answer = InternChatClient(api_key="test-key", retry=1).stream_chat([])
    assert next(answer) == "partial"
    with pytest.raises(RuntimeError):
        list(answer)


def test_stream_can_retry_before_first_content_and_finish_without_done(monkeypatch):
    responses = iter([
        Response([requests.ConnectionError("not connected")]),
        Response([frame("answer"), frame(reason="stop")]),
    ])
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr("llm_client.time.sleep", lambda _: None)
    assert list(InternChatClient(api_key="test-key", retry=2).stream_chat([])) == ["answer"]


@pytest.mark.parametrize("base", [
    "https://chat.intern-ai.org.cn/api/v1",
    "https://chat.intern-ai.org.cn/api/v1/",
    "https://chat.intern-ai.org.cn/api/v1/chat/completions",
    " https://chat.intern-ai.org.cn/api/v1/chat/completions/ ",
])
def test_api_base_accepts_root_and_complete_endpoint(monkeypatch, base):
    monkeypatch.setenv("INTERN_API_BASE", base)
    response = Response([frame("answer"), b"data: [DONE]"])
    urls = []

    def post(url, **kwargs):
        urls.append(url)
        return response

    monkeypatch.setattr(requests, "post", post)
    assert list(InternChatClient(api_key="test-key", retry=1).stream_chat([])) == ["answer"]
    assert urls == ["https://chat.intern-ai.org.cn/api/v1/chat/completions"]


def test_connect_timeout_retries_once_and_reports_readable_failure(monkeypatch):
    timeouts = []

    def post(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        raise requests.ConnectTimeout("sensitive low-level connection details")

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr("llm_client.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="连接模型服务超时") as error:
        list(InternChatClient(api_key="test-key", timeout=280, retry=2).stream_chat([]))
    assert timeouts == [(10, 280), (10, 280)]
    assert "sensitive" not in str(error.value)


def test_success_after_connect_timeout_keeps_only_successful_answer(monkeypatch):
    calls = []

    def post(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise requests.ConnectTimeout("temporary network failure")
        return Response([frame("answer"), b"data: [DONE]"])

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr("llm_client.time.sleep", lambda _: None)
    assert list(InternChatClient(api_key="test-key", retry=2).stream_chat([])) == ["answer"]
    assert len(calls) == 2


def test_non_transient_stream_error_is_not_retried(monkeypatch):
    response = Response([b'data: {"error":{"message":"invalid API key"}}'])
    calls = []

    def post(*args, **kwargs):
        calls.append(True)
        return response

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(RuntimeError, match="invalid API key"):
        list(InternChatClient(api_key="test-key", retry=2).stream_chat([]))
    assert len(calls) == 1
