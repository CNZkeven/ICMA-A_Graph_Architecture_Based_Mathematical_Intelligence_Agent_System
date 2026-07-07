from typing import Any, Dict

from langgraph_math_agent import MathAgentGraph
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
        self.skills_loader = SkillsLoader()
        self.mcp_client = PythonMCPClient()
        self.graph = MathAgentGraph(
            client=client, skills_loader=self.skills_loader, mcp_client=self.mcp_client)
        self.logger.info("ReasoningAgent initialized")

    def solve(self, problem: str, metadata: dict) -> dict:
        idx = metadata.get("idx", -1)
        try:
            self.logger.info(f"Solving problem {idx}")
            initial = create_initial_state(problem, metadata)
            final_state = self.graph.run(initial, token_budget=TokenBudget())
            result = {
                "final_response": final_state.get("final_response", "无法生成答案"),
                "trace": self._build_trace(final_state),
            }
            self._validate_output(result)
            return result
        except Exception as e:
            self.logger.error(f"Error solving problem {idx}: {e}")
            return {
                "final_response": "解题过程中出现错误，无法给出完整答案。",
                "trace": [{"step": "error", "content": str(e), "idx": idx}],
            }

    def _build_trace(self, state: dict) -> list:
        trace = []
        if state.get("category"):
            trace.append({
                "step": "classification",
                "category": state.get("category"),
                "confidence": state.get("category_confidence", 0.0),
                "candidates": state.get("candidate_categories", []),
                "tools": [
                    {
                        "name": "category_candidates",
                        "stages": state.get("classification_stages_used", []),
                        "selected": state.get("category"),
                    }
                ],
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
                    "final_answer": rr.get("answer", ""),
                    "validation_points": rr.get("validation_points", []),
                },
                "tools": [
                    {
                        "name": "skill_document",
                        "category": state.get("category", ""),
                        "purpose": "加载对应数学领域 skill 文档作为解题参考。",
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
                    "code_summary": self._truncate(python_code, 1600),
                    "expected_output_marker": "最终答案:",
                    "extracted_answer": po.get("answer", ""),
                },
                "tools": [
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
                        "stdout": self._truncate((po.get("stdout") or "").strip(), 1000),
                        "stderr": self._truncate((po.get("stderr") or "").strip(), 1000),
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
                },
                "tools": [
                    {
                        "name": "answer_matcher",
                        "details": state.get("validation_details", {}),
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
                    },
                    {
                        "name": "answer_formatter",
                        "final_response_length": len(state.get("final_response", "")),
                    },
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

    def _validate_output(self, result: dict) -> None:
        import json
        assert isinstance(result, dict), "返回值必须是dict"
        assert "final_response" in result, "必须包含final_response"
        assert isinstance(result["final_response"], str), "final_response必须是字符串"
        assert result["final_response"].strip(), "final_response不能为空"
        json.dumps(result, ensure_ascii=False)  # must be JSON-serializable


# ===================== PARTICIPANT DESIGN AREA END =====================
