"""Database retrieval node: fetch the top-k similar problems+solutions.

Runs before reasoning_agent and python_agent to provide reference examples.
"""

import os
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config import CONFIG
from utils.deps import get_deps
from utils.logger import get_logger
from utils.reference_block import partition_reference_examples


def _empty_retrieval_result() -> Dict[str, Any]:
    """Return the complete branch contract when retrieval is unavailable."""
    partition = partition_reference_examples([])
    return {
        "retrieved_examples": [],
        "reasoning_references": partition["reasoning"],
        "python_references": partition["python"],
        "reference_partition": partition,
    }


def database_retrieval_node(state: Dict[str, Any], config: Dict) -> Dict[str, Any]:
    """Query ChromaDB for the top-k similar problems+solutions as reference examples.

    k comes from ``CONFIG["db_retrieval_top_k"]`` (2). Every retrieved example is
    injected into both the reasoning and the Python agent.

    Returns:
        Dict with 'retrieved_examples': List[Dict] containing:
            - problem: original problem text
            - solution: solution text
            - similarity: cosine similarity score
            - source: source file path
    """
    # 检索是纯增益节点：任何失败都必须降级为"无参考示例"，绝不能打断求解子图。
    # get_deps 曾在 try 之外——缺 deps 的配置会让整条子图记下 KeyError（2026-08-19
    # 由 test_solving_subgraph 暴露）。
    try:
        deps = get_deps(config)
        logger = deps.logger or get_logger()
    except Exception:  # noqa: BLE001 - 配置形状是调用方的事，检索不为此失败。
        get_logger().warning("[db_retrieval] deps unavailable, skip")
        return _empty_retrieval_result()

    problem = state.get("problem", "").strip()
    if not problem:
        logger.warning("[db_retrieval] empty problem, skip")
        return _empty_retrieval_result()

    try:
        db_client = deps.db_client
        if db_client is None:
            logger.warning("[db_retrieval] db_client not initialized, skip")
            return _empty_retrieval_result()

        top_k = int(CONFIG.get("db_retrieval_top_k", 2) or 2)
        results = db_client.query(problem, top_k=top_k)
        # 相似度门控（2026-08-31 评委意见改进点 1）：低于阈值的近邻与本题结构
        # 相差过大（实测 sim≈0.44 的"路径计数"近邻把两分支同时带偏），注入的
        # 误导风险高于方法参考价值——不足即弃，宁缺毋滥。
        min_sim = float(CONFIG.get("db_reference_min_similarity", 0.55) or 0.0)
        dropped = []
        examples = []
        for i, r in enumerate(results):
            sim = float(r.get("similarity", 0.0) or 0.0)
            if sim < min_sim:
                dropped.append((i + 1, round(sim, 3), str(r.get("source", ""))[:60]))
                continue
            examples.append({
                "problem": r.get("problem", ""),
                "solution": r.get("solution", ""),
                "similarity": sim,
                "source": r.get("source", ""),
            })
            logger.info(
                f"[db_retrieval] #{i+1} sim={sim:.3f} "
                f"src={r.get('source', '')[:80]}"
            )
        for rank, sim, src in dropped:
            logger.info(f"[db_retrieval] dropped #{rank} sim={sim:.3f} below {min_sim} src={src}")

        partition = partition_reference_examples(examples)
        logger.info(
            f"[db_retrieval] retrieved {len(examples)}/{top_k} examples; "
            f"reasoning={len(partition['reasoning'])}, "
            f"python={len(partition['python'])} (solution text suppressed)"
        )
        return {
            "retrieved_examples": examples,
            "reasoning_references": partition["reasoning"],
            "python_references": partition["python"],
            "reference_partition": partition,
        }

    except Exception as e:
        logger.error(f"[db_retrieval] error: {e!r}")
        return _empty_retrieval_result()
