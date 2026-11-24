"""Staged replays of recorded real ICMA runs (the "试试竞赛题" home cards).

Each curated competition example ships with a recording made by
``record_staged.py``: the real pipeline's stage events and final answer.  A
staged run re-emits those events with demo pacing -- the reasoning and Python
agents start together and each finishes after 50-70 s, every other stage after
0.8-3 s -- and reports the paced durations as the stages' elapsed times, so
the ticking timer, the stage list and the persisted trace all agree.

Stage payloads are replayed unchanged; only their timing is staged.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .fixtures import COMPETITION_EXAMPLES


STAGED_DIR = Path(__file__).resolve().parent / "staged"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Recordings store project paths (skill documents, validation scripts) relative
# to this token so they stay machine-independent; replays restore the local root.
ROOT_TOKEN = "{ICMA_ROOT}"
AGENT_NODES = frozenset({"reasoning_agent", "python_agent"})
AGENT_SECONDS = (50.0, 70.0)
STAGE_SECONDS = (0.8, 3.0)

_LAYOUT_MACROS = re.compile(
    r"\\(?:displaystyle|textstyle|left|right|quad|qquad)(?![A-Za-z])|\\[,;:! ]"
)
_PRESENTATION = re.compile(r"[\s$,;:!?。、]")


def normalize_problem(text: str) -> str:
    """Canonical form used to recognise a pasted or retyped card problem.

    Only presentation differences are ignored: whitespace, ``$`` delimiters,
    LaTeX spacing/sizing macros and sentence punctuation.  Any change to the
    mathematics -- a coefficient, exponent or bound -- yields a different
    string, so a modified problem always goes to the real solver.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\\mathrm{d}", "d").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = _LAYOUT_MACROS.sub("", text)
    text = _PRESENTATION.sub("", text)
    return text.rstrip(".")


def match_staged_example(message: str) -> dict[str, Any] | None:
    """The competition example whose problem ``message`` restates, if any."""
    wanted = normalize_problem(message)
    if not wanted:
        return None
    for example in COMPETITION_EXAMPLES:
        variants = [example["problem"], *example.get("aliases", [])]
        if any(normalize_problem(variant) == wanted for variant in variants):
            return example
    return None


def rebase_paths(value: Any, old: str, new: str) -> Any:
    """Replace the ``old`` path prefix with ``new`` in every string of ``value``."""
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [rebase_paths(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: rebase_paths(item, old, new) for key, item in value.items()}
    return value


@lru_cache(maxsize=None)
def load_recording(example_id: str) -> dict[str, Any]:
    data = json.loads((STAGED_DIR / f"{example_id}.json").read_text(encoding="utf-8"))
    return rebase_paths(data, ROOT_TOKEN, str(PROJECT_ROOT))


def _executions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair every recorded ``node.started`` with the event that finished it."""
    executions: list[dict[str, Any]] = []
    open_runs: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        node, kind = event.get("node"), str(event.get("type") or "")
        if not node or not kind.startswith("node."):
            continue
        if kind == "node.started":
            run = {"node": node, "start": float(event.get("t") or 0), "started": event}
            open_runs.setdefault(node, []).append(run)
            executions.append(run)
        elif open_runs.get(node):
            run = open_runs[node].pop(0)
            run["end"] = float(event.get("t") or 0)
            run["finished"] = event
    return [run for run in executions if "finished" in run]


def plan_phases(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group recorded node runs into phases; runs that overlapped share one.

    The real solving subgraph fans out reasoning and Python verification in
    parallel, so their recorded intervals overlap and they form one phase.
    Every other stage runs on its own.
    """
    phases: list[dict[str, Any]] = []
    for run in sorted(_executions(events), key=lambda item: item["start"]):
        if phases and run["start"] < phases[-1]["end"]:
            phases[-1]["runs"].append(run)
            phases[-1]["end"] = max(phases[-1]["end"], run["end"])
        else:
            phases.append({"runs": [run], "end": run["end"]})
    return [phase["runs"] for phase in phases]


def paced_seconds(node: str, rng: random.Random) -> float:
    low, high = AGENT_SECONDS if node in AGENT_NODES else STAGE_SECONDS
    return round(rng.uniform(low, high), 2)


VALIDATION_LABELS = {
    "match": "推理与验证一致",
    "mismatch": "推理与验证存在分歧",
    "mismatch_reconciling": "分歧，重跑推理与验证",
    "mismatch_arbitrating": "分歧，转语义仲裁",
    "uncertain_reconciling": "证据不足，重跑推理与验证",
    "uncertain_arbitrating": "证据不足，转语义仲裁",
}
SOURCE_LABELS = {"reasoning": "推理", "python": "Python", "objective_review": "客观复核"}


def display_summary(node: str, payload: dict[str, Any]) -> str:
    """A reader-facing one-liner for a stage, derived from its recorded payload.

    The pipeline's own completion summary is ``f"{node} 完成"``, which reads as
    an internal identifier on screen; this says what the stage found instead.
    """
    thinking = payload.get("thinking") or {}
    if node == "input":
        return "题目已接收并建立 problem anchor"
    if node == "classifier" and thinking.get("category"):
        return f"判定领域：{thinking['category']}"
    if node == "database_retrieval" and isinstance(thinking.get("reference_count"), int):
        return f"检索到 {thinking['reference_count']} 条相似题参考"
    if node == "reasoning_agent" and thinking.get("steps"):
        return f"完成 {len(thinking['steps'])} 步推导，形成候选结论"
    if node == "python_agent":
        if thinking.get("verification_trusted") and thinking.get("evidence_status") == "support":
            return "Python 独立复核通过"
        if thinking.get("evidence_status"):
            return f"Python 复核：{thinking['evidence_status']}"
    if node == "objective_review":
        return "客观题独立复核完成"
    if node == "cross_validator":
        decision = thinking.get("decision")
        return VALIDATION_LABELS.get(decision, f"交叉验证：{decision}" if decision else "交叉验证完成")
    if node == "reconciliation":
        return "分歧处理完成"
    if node == "semantic_arbiter":
        decision = thinking.get("decision")
        return f"语义仲裁：选定{SOURCE_LABELS.get(decision, decision)}候选" if decision else "语义仲裁完成"
    if node == "coordinator":
        return "整理为最终回答"
    return str(payload.get("summary") or node)


def run_staged(
    example: dict[str, Any],
    sink: Any,
    rng: random.Random | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Replay ``example``'s recording through ``sink`` with demo pacing."""
    rng = rng or random.Random()
    # Tests shrink the real waits; the reported durations stay nominal.
    scale = float(os.environ.get("ICMA_STAGED_TIME_SCALE", "1"))
    recording = load_recording(example["id"])
    trace: list[dict[str, Any]] = []
    for runs in plan_phases(recording["events"]):
        paced = [(run, paced_seconds(run["node"], rng)) for run in runs]
        for run, _ in paced:
            sink.emit("node.started", node=run["node"], payload=dict(run["started"].get("payload") or {}))
        finished_entries: dict[int, dict[str, Any]] = {}
        waited = 0.0
        for run, seconds in sorted(paced, key=lambda pair: pair[1]):
            sleep(max(0.0, seconds - waited) * scale)
            waited = seconds
            finished = run["finished"]
            payload = {**(finished.get("payload") or {}), "elapsed_s": seconds}
            payload["summary"] = display_summary(run["node"], payload)
            sink.emit(finished["type"], node=run["node"], payload=payload)
            finished_entries[id(run)] = {
                "step": run["node"],
                "status": "degraded" if finished["type"] == "node.degraded" else "completed",
                "content": payload.get("summary", ""),
                "elapsed_s": seconds,
                "thinking": payload.get("thinking", {}),
                "tools": payload.get("tools", []),
            }
        # Keep start order, which is the order the stage list showed them in.
        trace.extend(finished_entries[id(run)] for run, _ in paced)
    return {
        "final_response": example.get("answer_markdown") or recording["result"]["final_response"],
        "trace": trace,
        "metadata": {"staged": True, "example_id": example["id"]},
    }
