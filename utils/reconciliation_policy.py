"""Shared reconciliation-round policy for validation and retry routing.

A reconciliation round re-runs the whole solving subgraph — another reasoning call
plus another Python call. On this model that is the most expensive action available
(a single reasoning call measured 77-116s on a mid-difficulty problem, and hard
problems run longer), so the round limit is gated by both budgets: tokens, and now
wall clock. Buying a rerun we cannot finish is strictly worse than answering with
what we already have.
"""

from config import CONFIG
from utils.deps import get_deps
from utils.retry_affordability import DEFAULT_ATTEMPT_COST_S, last_attempt_cost


def _deps(config):
    try:
        return get_deps(config)
    except Exception:  # noqa: BLE001 - callers may pass a bare config in tests.
        return None


def reconciliation_round_limit(config) -> int:
    deps = _deps(config)
    budget = getattr(deps, "token_budget", None)
    clock = getattr(deps, "time_budget", None)
    if clock is not None and clock.fast_path():
        # No room for another full solving pass; allow only the round already done.
        return 1
    if budget and budget.is_tight():
        return 1
    return CONFIG["reconciliation_max_rounds"]


def projected_solving_seconds(time_budget) -> float:
    """Estimate one more parallel reasoning/Python pass from observed costs."""
    if time_budget is None:
        return 0.0
    return max(
        last_attempt_cost(time_budget, "reasoning"),
        last_attempt_cost(time_budget, "python"),
        DEFAULT_ATTEMPT_COST_S,
    )


def reconciliation_retry_available(state, config, *, force: bool = False) -> bool:
    """Whether entering reconciliation can still produce another solving run.

    ``force`` is used once for an unresolved evidence gap.  It may consume the
    hard-clock reserve, but never bypasses the circuit breaker or the fast path.
    The default path is intentionally unchanged for existing callers.
    """
    completed_rounds = state.get("reconciliation_round", 0)
    if completed_rounds + 1 >= reconciliation_round_limit(config):
        return False
    if force and state.get("forced_recheck_used"):
        return False
    deps = _deps(config)
    clock = getattr(deps, "time_budget", None)
    if clock is None:
        return True
    if clock.fast_path():
        return False
    projected = projected_solving_seconds(clock)
    if force:
        reserve = float(CONFIG.get("forced_recheck_reserve_s", 240))
        hard_remaining = getattr(clock, "remaining_hard", clock.remaining)()
        return hard_remaining >= projected + reserve
    return clock.can_afford(projected)
