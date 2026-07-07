from utils.deps import get_deps
from utils.answer_formatter import post_process_final_response
from utils.cot_stripper import is_placeholder_answer, strip_cot_prefix
from utils.llm_retry import chat_with_retry
from utils.prompt_templates import COORDINATOR_PROMPT
from utils.token_budget import estimate_tokens
from config import CONFIG


def coordinator_node(state, config):
    deps = get_deps(config)
    client = deps.client
    budget = deps.token_budget
    rr = state.get("reasoning_result") or {}
    validated = state.get("validated_answer") or rr.get("answer", "")
    if is_placeholder_answer(validated):
        validated = ""
    ptype = (state.get("validation_details") or {}).get("problem_type", "computation")
    problem_type_label = "证明题" if ptype == "proof" else "计算题"
    steps_fmt = "\n".join(f"步骤{s.get('step_num')}: {s.get('description', '')}" for s in rr.get("steps", []))
    prompt = COORDINATOR_PROMPT.format(
        problem=state["problem"], category=state.get("category", ""),
        problem_type_label=problem_type_label,
        reasoning_steps_formatted=steps_fmt,
        reasoning_analysis=rr.get("analysis", ""),
        python_code=state.get("python_code", ""),
        python_output=(state.get("python_output") or {}).get("stdout", ""),
        validation_status=state.get("validation_status", ""),
        validated_answer=validated)
    raw = chat_with_retry(
        client,
        messages=[{"role": "user", "content": prompt}],
        temperature=CONFIG["temperatures"]["coordinator"],
        max_tokens=CONFIG["max_tokens"]["coordinator"],
        logger=deps.logger,
    )
    if budget:
        budget.consume(estimate_tokens(prompt), estimate_tokens(raw))
    cleaned_raw = strip_cot_prefix(raw)
    final = post_process_final_response(cleaned_raw, validated, ptype, problem=state["problem"])
    if not isinstance(final, str) or not final.strip():
        final = f"最终答案：{validated}" if validated else "无法生成完整答案。"
    # coordination_detail 保留完整解题说明，供 trace 记录（计算题 final_response 仅含简洁答案）
    return {"final_response": final, "coordination_detail": cleaned_raw}
