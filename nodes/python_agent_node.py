"""Python Agent: generate Python verification code, execute via MCP, self-correct. §3.4.

M4 hardening: intern-s2-preview emits a 'Thinking Process:' CoT and sometimes
writes '```python' mid-sentence in prose. _extract_code uses a strict fence
(newline-required) + content-based fallback, and returns '' (→ retry) when no
code is found — never executes CoT as code.
"""
import re
from utils.deps import get_deps
from utils.llm_retry import chat_with_retry
from utils.prompt_templates import PYTHON_PROMPT
from utils.token_budget import estimate_tokens
from utils.cot_stripper import strip_cot_prefix
from config import CONFIG


def _extract_code(response: str) -> str:
    """Extract Python code from an LLM response.

    1. strict fence (```python on its own line, newline required) — avoids
       matching '```python code block' written mid-sentence in CoT prose.
    2. loose fence, only if the content looks like code.
    3. content-based: first 'import'/'from' line to end (stop at closing ```).
    4. return '' if nothing code-like — do NOT execute CoT as code.
    """
    response = strip_cot_prefix(response)
    m = re.search(r"```python[ \t]*\n(.*?)\n```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```python\s+(.*?)\s+```", response, re.DOTALL)
    if m:
        cand = m.group(1).strip()
        if re.match(r"^\s*(import |from |#|def |class |if |for |while |try |with |@[a-zA-Z]|[a-zA-Z_]\w*\s*=)", cand):
            return cand
    m = re.search(r"```\s+(.*?)\s+```", response, re.DOTALL)
    if m:
        cand = m.group(1).strip()
        if re.match(r"^\s*(import |from |#|def )", cand):
            return cand
    lines = response.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(import |from )", ln):
            start = i
            break
    if start is not None:
        block = []
        for ln in lines[start:]:
            if ln.strip() == "```":
                break
            block.append(ln)
        return "\n".join(block).strip()
    return ""


def python_agent_node(state, config):
    deps = get_deps(config)
    sl = deps.skills_loader
    client = deps.client
    mcp_client = deps.mcp_client
    budget = deps.token_budget
    max_attempts = 1 if budget and budget.is_tight() else CONFIG["max_retries_per_node"]
    problem, category = state["problem"], state["category"]
    try:
        validation_script = sl.get_validation_script(category)
    except Exception:
        validation_script = ""
    base_prompt = PYTHON_PROMPT.format(
        validation_script=validation_script[:2000], problem=problem, category=category)
    hint = state.get("branch_hint")
    prompt = f"{hint}\n\n问题：{problem}" if hint else base_prompt

    trace = []
    attempts = 0
    last_output = None
    last_code = ""
    for _ in range(max_attempts):
        attempts += 1
        resp = chat_with_retry(
            client,
            messages=[{"role": "user", "content": prompt}],
            temperature=CONFIG["temperatures"]["python"],
            max_tokens=CONFIG["max_tokens"]["python"],
            logger=deps.logger,
        )
        if budget:
            budget.consume(estimate_tokens(prompt), estimate_tokens(resp))
        code = _extract_code(resp)
        last_code = code
        if not code:
            last_output = {"success": False, "stdout": "", "stderr": "no code block extracted",
                           "answer": None, "execution_time": 0.0}
            trace.append({"attempt": attempts, "status": "failed", "error": "no code block"})
            prompt = ("你上一次没有输出 ```python``` 代码块。请只输出一个 ```python``` 代码块，"
                      "不要任何其他文字。用 sympy 计算，最后一行 print(\"最终答案:\", answer)。\n问题：" + problem)
            continue
        if mcp_client is None:
            output = {"success": False, "stdout": "", "stderr": "MCP client unavailable",
                      "answer": None, "execution_time": 0.0}
        else:
            output = mcp_client.execute(code, timeout=CONFIG["node_timeouts"]["python_mcp_execute"])
        last_output = output
        trace.append({
            "attempt": attempts,
            "status": "success" if output.get("success") else "failed",
            "error": (output.get("stderr") or "")[:200],
        })
        if output.get("success"):
            return {"python_code": code, "python_output": output,
                    "python_trace": trace, "python_attempts": attempts}
        prompt = (
            f"代码执行失败：\n错误：{(output.get('stderr') or '')[:500]}\n"
            f"请只输出修正后的 ```python``` 代码块，用 sympy，结尾 print(\"最终答案:\", answer)。\n问题：{problem}"
        )
    return {"python_code": last_code, "python_output": last_output,
            "python_trace": trace, "python_attempts": attempts}
