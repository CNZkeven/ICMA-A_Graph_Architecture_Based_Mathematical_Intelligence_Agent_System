"""Local MCP server: execute Python code for math verification.

`execute_python` is the MCP tool impl (subprocess-based, isolated). A FastMCP
server wraps it for standalone MCP use; in-process callers import `execute_python`
directly. `_exec_in_process` is the no-subprocess fallback.
"""
import os
import re
import sys
import time
import tempfile
import subprocess
from typing import Optional


def _extract_answer_from_output(stdout: str) -> Optional[str]:
    if not stdout:
        return None
    # 取最后一个"最终答案:"标记到 stdout 末尾的全部内容——SymPy Matrix/子群列表等
    # 多行输出必须完整保留（单行截断曾导致 final_response 只剩 "(Matrix([" 碎片）。
    markers = list(re.finditer(r"最终答案[：:][ \t]*", stdout))
    if markers:
        answer = stdout[markers[-1].end():].strip()
        # 容错解码（errors="replace"）在截断处产生的 U+FFFD 只可能是残缺垃圾，
        # 从首尾剥掉，避免污染答案匹配。
        answer = answer.strip("�").strip()
        return answer if answer else None
    lines = [line.strip().strip("�").strip() for line in stdout.split("\n") if line.strip()]
    lines = [line for line in lines if line]
    return lines[-1] if lines else None


def execute_python(code: str, timeout: int = 30) -> dict:
    """MCP tool: run Python code in an isolated subprocess.

    Returns {success, stdout, stderr, answer, execution_time}.
    """
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            temp_file = f.name
        start = time.time()
        # Minimal env: PATH (find python), HOME (~/.local site-packages for sympy/numpy),
        # LANG/LC_ALL (UTF-8 for Chinese output). No secrets leaked.
        # PYTHONIOENCODING/PYTHONUTF8：Windows 上 LANG/LC_ALL 无效，子进程写管道默认用
        # ANSI 代码页——在 GBK 机器上与父进程的 UTF-8 解码必然错配，显式钉死 UTF-8。
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
        # errors="replace"：子进程被超时 kill/原生扩展崩溃时可能写到多字节字符中间，
        # 管道 EOF 落在残缺字节上；严格解码会让 capture_output 的 reader 线程抛
        # UnicodeDecodeError（subprocess.py _readerthread），整条 stdout 丢失。
        # 容错解码把残缺字节降级为 U+FFFD，已产出的"最终答案:"仍可抽取。
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace", env=env,
        )
        execution_time = time.time() - start
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "answer": _extract_answer_from_output(result.stdout),
            "execution_time": execution_time,
        }
    except subprocess.TimeoutExpired as e:
        # kill 后 subprocess.run 会把已回收的部分输出塞进异常对象；超时前打印的
        # "最终答案:"（如 LLM 代码先出答案再陷入死循环/长绘图）从这里救回。
        partial_stdout = e.stdout if isinstance(e.stdout, str) else (
            (e.stdout or b"").decode("utf-8", errors="replace") if e.stdout else "")
        return {"success": False, "stdout": partial_stdout,
                "stderr": f"代码执行超时（>{timeout}秒）",
                "answer": _extract_answer_from_output(partial_stdout),
                "execution_time": float(timeout)}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e),
                "answer": None, "execution_time": 0.0}
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass


def _exec_in_process(code: str, timeout: int = 30) -> dict:
    """Fallback: exec code in a restricted in-process namespace (no subprocess).

    Thread-based timeout (daemon thread; cannot hard-kill — acceptable for fallback).
    """
    import io
    import contextlib
    import threading

    namespace = {"__builtins__": __builtins__}
    for mod in ("sympy", "numpy", "math", "scipy"):
        try:
            namespace[mod] = __import__(mod)
        except Exception:
            pass

    buf = io.StringIO()
    done = {"result": None}

    def _run():
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, namespace)
            done["result"] = ("ok", buf.getvalue(), "")
        except Exception as e:
            done["result"] = ("err", buf.getvalue(), str(e))

    t = threading.Thread(target=_run, daemon=True)
    start = time.time()
    t.start()
    t.join(timeout)
    elapsed = time.time() - start
    if t.is_alive():
        return {"success": False, "stdout": buf.getvalue(),
                "stderr": f"代码执行超时（>{timeout}秒）",
                "answer": _extract_answer_from_output(buf.getvalue()),
                "execution_time": float(timeout)}
    status, stdout, stderr = done["result"]
    return {"success": status == "ok", "stdout": stdout, "stderr": stderr,
            "answer": _extract_answer_from_output(stdout), "execution_time": elapsed}


# ---- MCP server (standalone use) ----
def _build_mcp_server():
    from mcp.server.fastmcp import FastMCP
    server = FastMCP("python_executor")

    @server.tool()
    def execute_python_tool(code: str, timeout: int = 30) -> dict:
        """在隔离子进程中执行一段Python代码，返回stdout/stderr及抽取的最终答案。"""
        return execute_python(code, timeout)

    return server


try:
    mcp_server = _build_mcp_server()
except Exception:
    mcp_server = None


if __name__ == "__main__":
    if mcp_server is not None:
        mcp_server.run()
    else:
        print("FastMCP server unavailable; execute_python still callable as a function.")
