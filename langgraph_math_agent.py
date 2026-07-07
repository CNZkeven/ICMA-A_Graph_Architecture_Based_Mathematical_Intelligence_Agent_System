from langgraph.graph import StateGraph, START, END
from state.math_agent_state import MathAgentState
from graph.solving_subgraph import build_solving_subgraph
from nodes import input_node, classifier_node, reconciliation_node, coordinator_node
from utils.error_handler import node_wrapper
from config import CONFIG

def build_math_agent_graph():
    g = StateGraph(MathAgentState)
    g.add_node("input", node_wrapper(input_node, "input"))
    g.add_node("classifier", node_wrapper(classifier_node, "classifier"))
    g.add_node("solving", build_solving_subgraph())
    g.add_node("reconciliation", node_wrapper(reconciliation_node, "reconciliation"))
    g.add_node("coordinator", node_wrapper(coordinator_node, "coordinator"))

    g.add_edge(START, "input")
    g.add_edge("input", "classifier")
    g.add_edge("classifier", "solving")

    def route_after_solving(state, config):
        if state.get("should_terminate", False):
            return "coordinator"
        return "reconciliation" if state.get("next_node") == "reconciliation" else "coordinator"
    g.add_conditional_edges("solving", route_after_solving,
                            {"coordinator": "coordinator", "reconciliation": "reconciliation"})

    def route_after_reconciliation(state, config):
        if state.get("reconciliation_round", 0) >= CONFIG["reconciliation_max_rounds"]:
            return "coordinator"
        return "solving" if state.get("next_node") == "solving" else "coordinator"
    g.add_conditional_edges("reconciliation", route_after_reconciliation,
                            {"solving": "solving", "coordinator": "coordinator"})

    g.add_edge("coordinator", END)
    return g.compile()

class MathAgentGraph:
    def __init__(self, client, skills_loader=None, mcp_client=None):
        self.app = build_math_agent_graph()
        self.client = client
        self.skills_loader = skills_loader
        self.mcp_client = mcp_client

    def run(self, initial_state, token_budget=None):
        from utils.deps import Deps
        from utils.token_budget import TokenBudget
        tb = token_budget or TokenBudget()
        deps = Deps(client=self.client, skills_loader=self.skills_loader,
                    mcp_client=self.mcp_client, token_budget=tb)
        return self.app.invoke(initial_state, config={"configurable": {"deps": deps}})
