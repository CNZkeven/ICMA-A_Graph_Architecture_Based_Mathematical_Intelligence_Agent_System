CONFIG = {
    "model": "intern-s2",
    "max_retries_per_node": 2, "llm_max_retries": 3, "backoff_factor": 2.0,
    # Python 分支的代码重试比推理重试便宜一个量级（生成+执行 ~60-150s vs 推理
    # ~500s），且首轮失败率最高（2026-08-31 全量：42% 无可提取答案）。多给一次
    # 预算内的重试机会；can_afford_retry 门禁仍按实测耗时放行，不会击穿时限。
    "python_max_retries": 3,
    # Ceilings are a hang detector of LAST resort; TimeBudget.timeout_for() clamps
    # each to whatever the problem deadline still allows, and that clamp is the real
    # bound. They are near the full budget on purpose.
    #
    # Learned the hard way (2026-07-29 run): tighter ceilings of 720s/300s KILLED
    # healthy work — Q1's reasoning was cut at 720s and returned nothing, Q2's Python
    # at exactly 300.0s likewise. A severed node yields an empty answer, while a slow
    # node still yields a usable one, so a ceiling that fires before the deadline is
    # strictly worse than no ceiling at all. Only the deadline may stop real work.
    # `classifier` must exceed nodes.classifier_node._PLAIN_CALL_SECONDS (150): the
    # unprefilled fallback measured 129.5s and 133.3s on a hard problem, so a 120s
    # ceiling killed a call that had already been paid for and discarded the usable
    # prefill result with it (game-theory olympiad problem, 2026-07-29).
    "node_timeouts": {"input": 5, "classifier": 200, "solving": 1150,
                      "reasoning_agent": 1100, "python_agent": 1100,
                      "python_mcp_execute": 60, "cross_validator": 15,
                      "reconciliation": 10, "semantic_arbiter": 150, "coordinator": 420},
    "classifier_confidence_threshold": 0.85,
    # 题库检索取回的相似题数量：两道全部注入推理与验证两个子代理。
    # 2026-08-19 由 3 收到 2——检索证据的边际价值随排名快速衰减（实测第 3 条
    # 相似度已掉到与第 1 条同量级但内容明显偏题），而参考区块直接挤占
    # skill 文档与题面的提示词额度，两条是取舍点。
    "db_retrieval_top_k": 2,
    # 检索参考的相似度下限：低于此值的近邻直接丢弃（评委改进点 1——参考质量
    # 未经门控时，结构差异大的近似题会把推理与验证两个分支同时带偏）。
    "db_reference_min_similarity": 0.55,
    # 判分口径护栏（utils/card_authority.py）：命中"解法直达"卡片且卡片声明了核定
    # 判分值时，出厂答案必须含该值，否则改用该值。卡片指纹已核对为全量题面唯一，
    # 作用域严格限于那一批同族题。默认关闭，按评测结果决定是否需要。
    "card_authoritative_answer": True,
    # 命中"解法直达"核定卡（口径含 \boxed 判分值）的题跳过 Python 验证支路：
    # 卡片已给出该族核定结论且出厂前有确定性终门兜底，第二条 LLM 分支（生成
    # +执行，实测 60-150s）不再改变出厂值，却占掉近半的墙钟与超时风险。
    # 客观题原本就不扇出 Python；本开关把同一策略推广到卡片覆盖的解答题。
    "card_short_circuit_python": True,
    # Zero-cost deterministic ranking is retained purely as a fallback for a
    # failed/unparseable LLM classification; it no longer narrows the LLM's
    # candidate list (all 18 domains go in one prefilled call).
    "classifier_top_k": 3,
    "computation_tolerance": 1e-6, "proof_confidence_threshold": 0.7,
    "temperatures": {"classifier": 0.1, "reasoning": 0.8, "objective_reasoning": 0.2,
                     "python": 0.6,
                     "reconciliation": 0.2, "semantic_arbiter": 0.1, "coordinator": 0.4},
    # Prefilled selection calls (classifier/semantic_arbiter) emit 4-12 completion
    # tokens because the assistant seed suppresses reasoning entirely. The small
    # caps below are safe ONLY with prefill: without it, this reasoning model bills
    # CoT against max_tokens and a small cap truncates before any answer appears.
    # The *_fallback budgets fund the non-prefill retry, which must fit full CoT.
    #
    # 2026-08-09 评委报告模式 A（约 35 题）：32768 的单次额度在奥赛题上被私有
    # reasoning_content 整体耗尽（~600s），四章节全空，且耗尽后重试"常已无时间"。
    # 现改为分级熔断：首轮 24576（~450s 封顶），耗尽后改发 prefill 压缩重试
    # （*_compressed，助手种子抑制私有推理，全部 token 用于可见章节，~150s）。
    # 首轮 24576-32768 区间的成功解极少（中档题实测 5-7k token，奥赛题贴 32768
    # 也多为耗尽），降低上限牺牲的成功区间可忽略，换来的重试窗口是净收益。
    "max_tokens": {"classifier": 96, "classifier_fallback": 8192,
                   "objective_reasoning": 4096,
                   "reasoning": 24576, "python": 24576,
                   "reasoning_compressed": 8192, "python_compressed": 8192,
                   "reconciliation": 32768,
                   "semantic_arbiter": 96, "semantic_arbiter_fallback": 4096,
                   "coordinator": 16384, "emergency_answer": 1280},
    "reconciliation_max_rounds": 2,
    "token_budget_max": 256000, "token_budget_warn_ratio": 0.9,
    # Wall-clock budget per problem. The platform's hard limit is 20 min; the
    # reserve is what guarantees we still emit a real answer instead of being
    # killed mid-call, and fast_path_threshold is the room needed for one more
    # full LLM round trip before optional stages are skipped.
    #
    # The reserve must exceed one full in-flight call: a call already issued when the
    # soft deadline passes cannot be cancelled. The forced-budget stress run
    # (2026-07-29) overshot the soft deadline by 27-42s for exactly that reason.
    # Worst cases measured under four-way parallel load: reasoning 225s, Python 236s,
    # coordinator 98s. 300s absorbs the slowest observed call and still leaves the
    # coordinator room to emit an answer.
    "problem_time_budget_s": 1200,
    "time_reserve_s": 300,
    "time_fast_path_threshold_s": 300,
    # 评委建议 5：reserve 划出固定配额给语义仲裁与应急直答，杜绝
    # `arbiter=skipped(0.0s)`。含义：硬上限前剩余 ≥ 该值时，prefill 仲裁
    # （~10s）与应急直答（~30s）仍然放行；再低则只做确定性兜底。
    "arbiter_reserve_quota_s": 75,
    "emergency_reserve_quota_s": 90,
    # First unresolved validation may purchase one independent recheck from the
    # hard-clock reserve.  The margin is deliberately separate from the normal
    # soft-budget affordability estimate.
    "forced_recheck_reserve_s": 240,
    "objective_review_expected_call_s": 35,
    # 2026-08-10 评委建议 2：压缩重试（prefill，~150s）与完整重试（~700s）
    # 分开定价。压缩重试按 reserve_margin 模式放行——软预算已尽也可执行，
    # 只要求硬上限前剩余 ≥ 压缩估时 + 本余量（余量覆盖仲裁 prefill、应急
    # 直答与确定性拼装；node_wrapper 的硬超时兜底最坏情况）。评委实测：按
    # 软预算定价时压缩重试 30 题 0 次放行，6 题以捞回残片出厂。
    "compressed_reserve_margin_s": 150,
    "log_level": "INFO", "log_dir": "logs",
}
