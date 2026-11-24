from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from state.math_agent_state import MathAgentState
from nodes import (
    reasoning_agent_node,
    objective_review_node,
    python_agent_node,
    cross_validator_node,
    database_retrieval_node,
)
from utils.error_handler import node_wrapper
from utils.problem_profile import is_objective_mode
from utils.card_authority import is_card_covered as _card_covered
from config import CONFIG

def build_solving_subgraph():
    sub = StateGraph(MathAgentState)
    sub.add_node("database_retrieval", node_wrapper(database_retrieval_node, "database_retrieval"))
    sub.add_node("reasoning_agent", node_wrapper(reasoning_agent_node, "reasoning_agent"))
    sub.add_node("objective_review", node_wrapper(objective_review_node, "objective_review"))
    sub.add_node("python_agent", node_wrapper(python_agent_node, "python_agent"))
    sub.add_node("cross_validator", node_wrapper(cross_validator_node, "cross_validator"))

    # Start with database retrieval
    sub.add_edge(START, "database_retrieval")

    def fan_out(state, config):
        reasoning_send = Send(
            "reasoning_agent", {**state, "branch_hint": state.get("reasoning_retry_hint")}
        )
        if is_objective_mode(state.get("question_mode", "computation")):
            # The objective reasoning node already performs its own concise
            # verification.  Do not fan out a second expensive LLM/code branch.
            return [reasoning_send]
        if CONFIG.get("card_short_circuit_python", False) and _card_covered(
                str(state.get("problem") or "")):
            # 命中核定卡：卡片把该族结论连同判分口径一起注入推理提示，出厂前还有
            # 确定性终门兜底，Python 分支不再改变出厂值——省掉近半墙钟与超时风险。
            return [reasoning_send]
        return [
            reasoning_send,
            Send("python_agent", {**state, "branch_hint": state.get("python_retry_hint")}),
        ]
    sub.add_conditional_edges("database_retrieval", fan_out, ["reasoning_agent", "python_agent"])
    def route_after_reasoning(state):
        return "objective_review" if is_objective_mode(
            state.get("question_mode", "computation")
        ) else "cross_validator"

    sub.add_conditional_edges(
        "reasoning_agent",
        route_after_reasoning,
        {"objective_review": "objective_review", "cross_validator": "cross_validator"},
    )
    sub.add_edge("objective_review", "cross_validator")
    sub.add_edge("python_agent", "cross_validator")
    sub.add_edge("cross_validator", END)
    return sub.compile()
