from typing import Annotated, Any, Dict, List, Optional
import operator
from typing_extensions import NotRequired, TypedDict

class MathAgentState(TypedDict):
    # 输入阶段（必需）
    problem: str
    metadata: Dict[str, Any]
    idx: int
    reasoning_attempts: int
    python_attempts: int
    errors: Annotated[List[Dict], operator.add]
    # 分类阶段
    category: NotRequired[str]
    category_confidence: NotRequired[float]
    candidate_categories: NotRequired[List[str]]
    classification_stages_used: NotRequired[List[str]]
    # 推理阶段
    reasoning_result: NotRequired[Optional[Dict[str, Any]]]
    reasoning_trace: NotRequired[List[Dict]]
    reasoning_retry_hint: NotRequired[Optional[str]]
    # Python 验证阶段
    python_code: NotRequired[str]
    python_output: NotRequired[Optional[Dict[str, Any]]]
    python_trace: NotRequired[List[Dict]]
    python_retry_hint: NotRequired[Optional[str]]
    # 验证阶段
    validation_status: NotRequired[str]
    validation_details: NotRequired[Dict[str, Any]]
    validated_answer: NotRequired[str]
    # 协调阶段
    reconciliation_trace: NotRequired[List[Dict]]
    reconciliation_round: NotRequired[int]
    # 输出阶段
    final_response: NotRequired[str]
    coordination_detail: NotRequired[Optional[str]]  # coordinator 原始解题说明，记入 trace（计算题 final_response 已收敛为简洁答案）
    # 内部控制
    branch_hint: NotRequired[Optional[str]]  # Send() 扇出携带的重试提示（solving 子图内）
    next_node: NotRequired[Optional[str]]
    should_terminate: NotRequired[bool]
    token_budget_consumed: NotRequired[int]

def create_initial_state(problem: str, metadata: Dict[str, Any]) -> MathAgentState:
    return MathAgentState(
        problem=problem, metadata=metadata, idx=metadata.get("idx", -1),
        reasoning_attempts=0, python_attempts=0, errors=[],
    )
