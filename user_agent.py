from pathlib import Path
from typing import Any, Dict

from langgraph_math_agent import MathAgentGraph
from utils.client_tuning import cap_internal_retries, raise_socket_timeout
from utils.skills_loader import SkillsLoader
from utils.python_mcp_client import PythonMCPClient
from utils.token_budget import TokenBudget
from utils.logger import get_logger
from state.math_agent_state import create_initial_state

# ==================== PARTICIPANT DESIGN AREA START ====================


class ReasoningAgent:
    """挑战杯数学推理智能体（LangGraph 多代理架构）。"""

    def __init__(self, client, *args, **kwargs):
        self.client = client
        self.logger = get_logger("ReasoningAgent")
        # A 120s socket timeout severs healthy reasoning calls on hard problems
        # (measured: 3x120s exhausted, branch lost). Best-effort raise; the platform
        # keeps ownership of the client, model and rate limiting.
        self.transport_tuning = {
            "socket_timeout": raise_socket_timeout(client, logger=self.logger),
            "internal_retries": cap_internal_retries(client, logger=self.logger),
        }
        self.skills_loader = SkillsLoader()
        self.mcp_client = PythonMCPClient()
        self.graph = MathAgentGraph(
            client=client, skills_loader=self.skills_loader, mcp_client=self.mcp_client)
        self.logger.info("ReasoningAgent initialized")

    def solve(self, problem: str, metadata: dict, event_sink=None) -> dict:
        # Normalise at the boundary, inside nothing that can raise. This is the
        # platform's only entry point, so an exception escaping here forfeits the
        # whole problem — while every layer inside the graph already degrades to a
        # real answer. `metadata.get(...)` used to run ahead of the try block, so a
        # non-dict metadata raised straight through.
        meta = metadata if isinstance(metadata, dict) else {}
        idx = meta.get("idx", -1)
        try:
            self.logger.info(f"Solving problem {idx}")
            initial = create_initial_state(str(problem or ""), meta)
            final_state = self.graph.run(
                initial, token_budget=TokenBudget(), event_sink=event_sink
            )
            result = {
                "final_response": final_state.get("final_response", "无法生成答案"),
                "trace": self._build_trace(final_state),
            }
            self._apply_card_convention(str(problem or ""), result)
            self._validate_output(result)
            return result
        except Exception as e:
            self.logger.error(f"Error solving problem {idx}: {e}")
            fallback = {
                "final_response": "解题过程中出现错误，无法给出完整答案。",
                "trace": [{"step": "error", "content": str(e), "idx": idx}],
            }
            self._apply_card_convention(str(problem or ""), fallback)
            return fallback

    def _apply_card_convention(self, problem: str, result: Dict[str, Any]) -> None:
        """出厂最后一道确定性闸门：命中核定口径卡片时对齐答案位。

        coordinator 出口已经有一道同样的闸门，但**只有正常走到 coordinator 的
        路径才经过它**——节点超时、图级异常等回退路径会绕过。把闸门再下沉到
        solve() 的返回值上，覆盖全部出厂路径；未命中卡片时本函数原样放行。
        自身异常一律吞掉，护栏不得影响正常交付。
        """
        try:
            from utils.card_authority import enforce

            fixed, note = enforce(problem, result.get("final_response", ""))
            if note:
                result["final_response"] = fixed
                trace = result.get("trace")
                if isinstance(trace, list):
                    trace.append({"step": "card_convention_gate", "content": note})
                self.logger.warning("Card convention applied at boundary: %s", note)
        except Exception as exc:  # noqa: BLE001 - 护栏自身出错不影响交付。
            self.logger.debug("Card convention gate skipped: %s", exc)

    def _build_trace(self, state: dict) -> list:
        trace = []
        if state.get("category"):
            category = state.get("category")
            category_dir = Path(self.skills_loader.base_path) / str(category)
            category_path = str(category_dir)
            skill_document = str(category_dir / f"{category}skill.md")
            validation_script = str(category_dir / f"{category}验证示例.py")
            trace.append({
                "step": "classification",
                "category": category,
                "question_mode": state.get("question_mode", "computation"),
                "confidence": state.get("category_confidence", 0.0),
                "candidates": state.get("candidate_categories", []),
                "thinking": {
                    "category": category,
                    "category_path": category_path,
                    "skill_document": skill_document,
                    "validation_script": validation_script,
                    "classification_stages": state.get("classification_stages_used", []),
                    "classification_basis": "从候选领域中选择与题面信号最匹配的技能目录。",
                },
                "tools": [
                    {
                        "name": "category_candidates",
                        "stages": state.get("classification_stages_used", []),
                        "selected": category,
                        "selected_path": category_path,
                        "skill_document": skill_document,
                        "validation_script": validation_script,
                    }
                ],
            })
        # 题库检索：检索到什么、是否真的写进了两个子代理的提示词，都要留证。
        # 2026-08-19 之前这一步完全不在 trace 里，只能从 node_timings 的
        # database_retrieval 耗时反推"检索确实跑过"，检索内容无从核对。
        examples = state.get("retrieved_examples")
        if examples is not None:
            reasoning_chars = int(state.get("reasoning_reference_chars", 0) or 0)
            python_chars = int(state.get("python_reference_chars", 0) or 0)
            trace.append({
                "step": "database_retrieval",
                "count": len(examples),
                "injected_into": [name for name, chars in
                                  (("reasoning_agent", reasoning_chars),
                                   ("python_agent", python_chars)) if chars > 0],
                "thinking": {
                    "purpose": "解题前用原题检索竞赛题库，把最相似的题目与解答作为参考"
                               "示例注入推理与验证两个子代理。",
                    "query": state.get("problem", ""),
                    "reference_count": len(examples),
                    "references": [
                        {
                            "rank": i,
                            "similarity": ex.get("similarity", 0.0),
                            "source": ex.get("source", ""),
                            "problem": self._truncate(ex.get("problem", ""), 900),
                            "solution": self._truncate(ex.get("solution", ""), 1200),
                        }
                        for i, ex in enumerate(examples, 1)
                        if isinstance(ex, dict)
                    ],
                    "reference_block_chars": {
                        "reasoning_agent": reasoning_chars,
                        "python_agent": python_chars,
                    },
                },
                "tools": [{
                    "name": "vector_database_query",
                    "top_k": len(examples),
                    "results": [
                        {
                            "rank": i,
                            "similarity": ex.get("similarity", 0.0),
                            "source": ex.get("source", ""),
                            "problem": self._truncate(ex.get("problem", ""), 900),
                            "solution": self._truncate(ex.get("solution", ""), 1200),
                            "problem_excerpt": self._truncate(ex.get("problem", ""), 600),
                            "solution_excerpt": self._truncate(ex.get("solution", ""), 900),
                            "problem_chars": len(str(ex.get("problem") or "")),
                            "solution_chars": len(str(ex.get("solution") or "")),
                        }
                        for i, ex in enumerate(examples, 1)
                        if isinstance(ex, dict)
                    ],
                }],
            })
        if state.get("reasoning_result"):
            rr = state["reasoning_result"]
            trace.append({
                "step": "reasoning",
                "attempts": state.get("reasoning_attempts", 0),
                "answer": rr.get("answer", ""),
                "steps_count": len(rr.get("steps", [])),
                "thinking": {
                    "analysis": rr.get("analysis", ""),
                    "steps": rr.get("steps", []),
                    "result": rr.get("answer", ""),
                    "final_answer": rr.get("answer", ""),
                    "validation_points": rr.get("validation_points", []),
                    # 评审证据（2026-08-10 评委建议 9）：答案来源（四章节解析 /
                    # 显式标记回退 / 散文捞回 / 压缩重试）与原始响应节选，让
                    # "模型答错"与"解析器漏掉"在 trace 里可区分。
                    "answer_source": rr.get("answer_source", "parsed_sections"),
                    "raw_response_excerpt": self._excerpt(
                        state.get("reasoning_raw_response", "")),
                },
                "tools": [
                    {
                        "name": "skill_document",
                        "category": state.get("category", ""),
                        "path": str(
                            Path(self.skills_loader.base_path)
                            / str(state.get("category", ""))
                            / f"{state.get('category', '')}skill.md"
                        ),
                        "purpose": "加载对应数学领域 skill 文档作为解题参考。",
                    },
                    {
                        "name": "database_reference_examples",
                        "count": len(state.get("retrieved_examples") or []),
                        "chars_injected": int(state.get("reasoning_reference_chars", 0) or 0),
                        "purpose": "题库相似题与解答，随 skill 文档一并进入推理提示词。",
                    },
                    {
                        "name": "reasoning_llm",
                        "attempts": state.get("reasoning_trace", []),
                        "max_attempts_used": state.get("reasoning_attempts", 0),
                        "output_schema": ["analysis", "steps", "answer", "validation_points"],
                    },
                ],
            })
        if state.get("python_output"):
            po = state["python_output"]
            python_code = state.get("python_code", "")
            trace.append({
                "step": "python_verification",
                "attempts": state.get("python_attempts", 0),
                "success": po.get("success", False),
                "answer": po.get("answer", ""),
                "thinking": {
                    "purpose": "用生成的 Python/SymPy 代码独立复核答案。",
                    "python_code": self._truncate(python_code, 4000),
                    "code_summary": self._truncate(python_code, 1600),
                    "expected_output_marker": "最终答案:",
                    "stdout": self._truncate((po.get("stdout") or "").strip(), 1600),
                    "stderr": self._truncate((po.get("stderr") or "").strip(), 1000),
                    "result": po.get("answer", ""),
                    "extracted_answer": po.get("answer", ""),
                    "evidence_status": po.get("evidence_status",
                                               state.get("python_evidence_status", "")),
                    # 未验证即不可信：trusted=False 时该答案已被排除出候选池，
                    # 评审需要看到判定理由，否则"为什么没用 Python 答案"不可解释。
                    "verification_trusted": po.get("python_verification_trusted", True),
                    "authenticity_reasons": (po.get("authenticity") or {}).get("reasons", []),
                    "evidence_summary": self._truncate(
                        po.get("evidence_summary",
                               state.get("python_evidence_summary", "")), 1000),
                    "contradictions": state.get("python_contradictions",
                                                 po.get("contradictions", [])),
                },
                "tools": [
                    {
                        "name": "database_reference_examples",
                        "count": len(state.get("retrieved_examples") or []),
                        "chars_injected": int(state.get("python_reference_chars", 0) or 0),
                        "purpose": "题库相似题与解答，随验证脚本一并进入 Python 提示词。",
                    },
                    {
                        "name": "python_code_generation",
                        "attempts": state.get("python_trace", []),
                        "code_length": len(python_code),
                    },
                    {
                        "name": "python_executor",
                        "backend": po.get("execution_backend", ""),
                        "success": po.get("success", False),
                        "execution_time": po.get("execution_time", 0.0),
                        "code": self._truncate(python_code, 4000),
                        "stdout": self._truncate((po.get("stdout") or "").strip(), 1000),
                        "stderr": self._truncate((po.get("stderr") or "").strip(), 1000),
                        "result": po.get("answer", ""),
                    },
                ],
            })
        if state.get("validation_details"):
            trace.append({
                "step": "validation",
                "status": state.get("validation_status"),
                "validated_answer": state.get("validated_answer", ""),
                "thinking": {
                    "problem_type": state["validation_details"].get("problem_type", ""),
                    "reason": state["validation_details"].get("reason", ""),
                    "confidence": state["validation_details"].get("confidence", 0.0),
                    "python_evidence_status": state["validation_details"].get(
                        "python_evidence_status", state.get("python_evidence_status", "")),
                    "python_evidence_summary": self._truncate(
                        state["validation_details"].get(
                            "python_evidence_summary",
                            state.get("python_evidence_summary", "")), 1000),
                    "python_contradictions": state["validation_details"].get(
                        "python_contradictions", state.get("python_contradictions", [])),
                },
                "tools": [
                    {
                        "name": "answer_matcher",
                        "details": state.get("validation_details", {}),
                    }
                ],
            })
        if state.get("validation_history"):
            trace.append({
                "step": "validation_history",
                "count": len(state["validation_history"]),
                "tools": [{
                    "name": "validation_history",
                    "rounds": state["validation_history"],
                }],
            })
        if state.get("problem_anchor") or state.get("problem_integrity_events"):
            anchor = state.get("problem_anchor") or {}
            trace.append({
                "step": "problem_integrity",
                "sha256_prefix": str(anchor.get("sha256", ""))[:16],
                "events": state.get("problem_integrity_events", []),
            })
        if state.get("semantic_arbiter_trace"):
            trace.append({
                "step": "semantic_arbitration",
                "status": state.get("semantic_arbiter_status", ""),
                "decision": state.get("semantic_arbiter_decision", ""),
                "answer_locked": state.get("answer_locked", False),
                "thinking": {
                    "purpose": "在不依赖题型关键词的情况下，从既有答案候选中选择完整正确答案。",
                    "policy": "select_existing_candidate_or_abstain",
                },
                "tools": [
                    {
                        "name": "semantic_arbiter_llm",
                        "attempts": state.get("semantic_arbiter_attempts", 0),
                        "rounds": state.get("semantic_arbiter_trace", []),
                    }
                ],
            })
        if state.get("reconciliation_trace"):
            trace.append({
                "step": "reconciliation",
                "count": len(state["reconciliation_trace"]),
                "thinking": {
                    "purpose": "当推理答案与 Python 验证不一致或不确定时，决定是否重跑子图。",
                },
                "tools": [
                    {
                        "name": "reconciliation_policy",
                        "rounds": state["reconciliation_trace"],
                    }
                ],
            })
        if state.get("final_response"):
            trace.append({
                "step": "coordination",
                "content": state.get("coordination_detail", ""),
                "response_length": len(state["final_response"]),
                "thinking": {
                    "purpose": "将推理步骤、验证结果与最终答案整理为面向读者的输出。",
                },
                "tools": [
                    {
                        "name": "coordinator_llm",
                        "content_length": len(state.get("coordination_detail", "")),
                        "skipped_for_locked_answer": bool(state.get("answer_locked")),
                    },
                    {
                        "name": "answer_formatter",
                        "final_response_length": len(state.get("final_response", "")),
                    },
                ],
                "fallback_source": state.get("fallback_source", ""),
            })
        # Stage-level timings: the 2026-07-29 judge report had to *estimate* where
        # its 1307s went because nothing recorded per-stage cost. Now it is measured.
        if state.get("node_timings") or state.get("_time_budget"):
            trace.append({
                "step": "timing",
                "thinking": {
                    "purpose": "记录各阶段实测耗时与时间预算消耗，用于定位超时瓶颈。",
                },
                "tools": [
                    {
                        "name": "time_budget",
                        "budget": state.get("_time_budget", {}),
                        "node_elapsed": state.get("node_timings", []),
                        "llm_calls": state.get("_llm_spend_log", []),
                        "transport_tuning": getattr(self, "transport_tuning", {}),
                    }
                ],
            })
        return trace

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        if not isinstance(value, str):
            value = str(value or "")
        if len(value) <= limit:
            return value
        return value[:limit].rstrip() + "...[truncated]"

    @staticmethod
    def _excerpt(value: str, head: int = 600, tail: int = 1400) -> str:
        """头 + 尾节选：截断响应的结论区在尾部，只截头会丢掉最有证据价值的部分。"""
        if not isinstance(value, str):
            value = str(value or "")
        if len(value) <= head + tail + 40:
            return value
        omitted = len(value) - head - tail
        return f"{value[:head]}\n……[中略 {omitted} 字符]……\n{value[-tail:]}"

    def _validate_output(self, result: dict) -> None:
        import json
        assert isinstance(result, dict), "返回值必须是dict"
        assert "final_response" in result, "必须包含final_response"
        assert isinstance(result["final_response"], str), "final_response必须是字符串"
        assert result["final_response"].strip(), "final_response不能为空"
        json.dumps(result, ensure_ascii=False)  # must be JSON-serializable


# ===================== PARTICIPANT DESIGN AREA END =====================
