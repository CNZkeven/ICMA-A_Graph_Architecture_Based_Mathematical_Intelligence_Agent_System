from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable, TypeVar

from config import CONFIG

T = TypeVar("T")

NODE_TIMEOUTS = dict(CONFIG["node_timeouts"])


class NodeTimeoutError(TimeoutError):
    pass


def get_node_timeout(node_name: str, default: int | None = None) -> int | None:
    return NODE_TIMEOUTS.get(node_name, default)


def run_with_timeout(func: Callable[[], T], timeout: int | None) -> T:
    if timeout is None:
        return func()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            raise NodeTimeoutError(f"operation timed out after {timeout}s") from exc
