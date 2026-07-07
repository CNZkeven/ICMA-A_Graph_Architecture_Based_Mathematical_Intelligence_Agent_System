"""Client for the python_executor MCP server (in-process call) + local fallback."""
from mcp_servers.python_executor import server as _server


class PythonMCPClient:
    """Calls execute_python (subprocess path = MCP tool) with exec-in-process fallback.

    The MCP server's tool function is called in-process (no MCP transport needed);
    this keeps the call synchronous and concurrency-safe (each call = independent
    subprocess). If the subprocess path raises, fall back to in-process exec.
    """

    def __init__(self, mcp_session=None):
        # mcp_session kept for interface parity with the plan; in-process mode ignores it.
        self.mcp_session = mcp_session

    def execute(self, code: str, timeout: int = 30) -> dict:
        try:
            result = _server.execute_python(code, timeout)
            result["execution_backend"] = "mcp"
            return result
        except Exception:
            result = _server._exec_in_process(code, timeout)
            result["execution_backend"] = "local_fallback"
            return result
