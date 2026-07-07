# 数学推理智能体（Math Reasoning Agent）

基于 LangGraph 的多代理数学解题系统，面向挑战杯竞赛。系统采用 **分类 → 双路并行求解 → 交叉验证 → 协调** 的流水线，结合 LLM 推理与 Python 符号计算（sympy）双路验证，输出带格式的最终答案。

---

## 一、架构

### 1.1 主图流水线

主图定义于 `langgraph_math_agent.py`，由 5 个节点组成：

```
START → input → classifier → solving(子图) → reconciliation ⇄ solving → coordinator → END
```

| 节点 | 文件 | 职责 |
|---|---|---|
| **input** | `nodes/input_node.py` | 提取 `idx`，初始化控制字段 |
| **classifier** | `nodes/classifier_node.py` | 三阶段分类：关键词检索 → TF-IDF 嵌入排序 → LLM 选定数学领域 |
| **solving** | `graph/solving_subgraph.py` | 子图：双路并行求解 + 交叉验证 |
| **reconciliation** | `nodes/reconciliation_node.py` | 答案不一致时生成重试提示，回到 solving（最多 2 轮） |
| **coordinator** | `nodes/coordinator_node.py` | 汇总推理步骤 + Python 输出 + 验证结果，生成并格式化最终响应 |

路由逻辑：
- `solving` 之后：若 `should_terminate` 或无需调解 → `coordinator`；否则 → `reconciliation`
- `reconciliation` 之后：若达到轮数上限 → `coordinator`；否则 → `solving` 重试

### 1.2 Solving 子图（双路扇出）

定义于 `graph/solving_subgraph.py`，使用 LangGraph `Send` 并行扇出：

```
START ──┬→ reasoning_agent ──┐
        └→ python_agent ─────┴→ cross_validator → END
```

| 节点 | 职责 |
|---|---|
| **reasoning_agent** | 加载该领域的 skill 文档，LLM 生成结构化推理（问题分析 / 详细解题步骤 / 最终答案 / 关键验证点），格式不全则重试 |
| **python_agent** | 加载该领域验证示例脚本，LLM 生成 sympy 代码 → 经 MCP 子进程执行 → 提取 `最终答案:` → 失败则自我修正 |
| **cross_validator** | `AnswerMatcher` 比对两路答案：计算题用数值/符号等价判定，证明题用推理完整性+Python 验证判定；输出 `match / mismatch / uncertain` |

### 1.3 模块布局

```
├── main.py                       # 竞赛入口：读 JSONL → 并发调用 → 写每题 JSON
├── user_agent.py                 # ReasoningAgent：竞赛接口，封装图运行 + trace 构建
├── langgraph_math_agent.py       # 主图构建（MathAgentGraph）
├── llm_client.py                 # InternChatClient（OpenAI 兼容 Chat 客户端）
├── config.py                     # 全局配置（模型/超时/温度/token 预算等）
├── requirements.txt
├── graph/solving_subgraph.py     # solving 子图（双路扇出）
├── nodes/                        # 7 个节点实现
│   ├── input_node.py
│   ├── classifier_node.py
│   ├── reasoning_agent_node.py
│   ├── python_agent_node.py
│   ├── cross_validator_node.py
│   ├── reconciliation_node.py
│   └── coordinator_node.py
├── state/math_agent_state.py     # MathAgentState（TypedDict，全图共享状态）
├── utils/                        # 工具模块
│   ├── deps.py                   # 依赖注入（client/skills_loader/mcp_client/token_budget）
│   ├── llm_retry.py              # 带重试的 LLM 调用
│   ├── skills_loader.py          # 加载 skill 文档 + 关键词/TF-IDF 检索
│   ├── category_embedding_index.py  # TF-IDF 嵌入索引
│   ├── python_mcp_client.py      # Python MCP 客户端（in-process 调用 + 本地回退）
│   ├── answer_matcher.py         # 答案匹配（数值/符号/字符串相似度）
│   ├── answer_extractor.py       # 答案归一化抽取
│   ├── answer_formatter.py       # 最终响应格式化（最终答案:/结论:）
│   ├── prompt_templates.py       # 各节点 Prompt 模板
│   ├── token_budget.py           # Token 预算管理
│   ├── timeout_control.py        # 超时控制
│   ├── error_handler.py          # 节点异常包装（node_wrapper）
│   └── logger.py                 # 日志
├── mcp_servers/python_executor/  # Python 代码执行 MCP server（隔离子进程）
└── skills_pythonscripts/         # 18 个数学领域 skill 文档 + 验证示例脚本
```

---

## 二、逻辑数据流

### 2.1 端到端流程

**输入**：JSONL 文件，每行 `{"idx": int, "problem": str, ...}`

```
main.py 加载 JSONL
   │  按 LOCAL_MAX_CONCURRENCY（默认 8）并发
   ▼
ReasoningAgent.solve(problem, metadata={idx})
   │  create_initial_state → MathAgentGraph.run
   ▼
┌─ input ─────────────── 写入 idx
│
├─ classifier ────────── 关键词检索候选（top_k=5）
│                       + TF-IDF 嵌入排序（union 去重，cap 7）
│                       → LLM 从候选中选定 category
│
├─ solving 子图（Send 并行扇出）─────────────────
│   ├─ reasoning_agent: 读 {category}skill.md → LLM 推理
│   │                  → 解析 {analysis, steps, answer, validation_points}
│   ├─ python_agent:    读 {category}验证示例.py → LLM 生成代码
│   │                  → MCP 子进程执行 → 抽取 最终答案: → 失败重试
│   └─ cross_validator: AnswerMatcher 比对
│                      → match / mismatch / uncertain + validated_answer
│
├─ reconciliation ────── （仅 mismatch/uncertain 且未达上限时）
│                       生成 retry_hint（指向失败的一方）→ 回 solving
│
└─ coordinator ───────── 汇总推理步骤 + Python 输出 + 验证状态
                        → LLM 生成最终响应
                        → post_process_final_response 格式化
   │
   ▼
ReasoningAgent 返回 {final_response, trace}
   │
   ▼
main.py 写入 {output_dir}/{idx}.json
```

### 2.2 状态流转

全图共享 `MathAgentState`（`state/math_agent_state.py`），关键字段：

- **输入**：`problem`, `idx`, `metadata`
- **分类**：`category`, `category_confidence`, `candidate_categories`
- **推理**：`reasoning_result`（含 analysis/steps/answer）, `reasoning_attempts`
- **Python**：`python_code`, `python_output`（含 success/stdout/stderr/answer）, `python_attempts`
- **验证**：`validation_status`, `validated_answer`, `validation_details`
- **调解**：`reconciliation_round`, `reconciliation_trace`, `reasoning_retry_hint`, `python_retry_hint`
- **控制**：`next_node`, `should_terminate`, `token_budget_consumed`
- **输出**：`final_response`

### 2.3 Token 预算机制

`TokenBudget`（默认上限 30000，告警比 0.8）跟踪所有 LLM 调用的 token 消耗。当消耗超过告警阈值进入 **tight 模式**：
- 分类跳过 TF-IDF 嵌入阶段（仅关键词）
- 各节点内部重试次数降为 1
- reconciliation 上限降为 1

### 2.4 答案选择策略

`cross_validator_node._preferred_answer`：
- **计算题 + Python 成功** → 采用 Python（sympy）答案（确定性优先）
- **证明题 / Python 失败** → 采用推理答案

---

## 三、使用方法

### 3.1 环境变量

| 变量 | 必需 | 默认值 | 说明 |
|---|---|---|---|
| `INTERN_API_KEY` | 是 | — | API 密钥（自动补 `Bearer ` 前缀） |
| `INTERN_API_BASE` | 否 | `https://chat.intern-ai.org.cn/api/v1/chat/completions` | API 端点 |
| `INTERN_MODEL` | 否 | `intern-s2-preview` | 模型名 |
| `LOCAL_MAX_CONCURRENCY` | 否 | `8` | 并发题数 |

### 3.2 安装

```bash
pip install -r requirements.txt
```

### 3.3 运行

```bash
export INTERN_API_KEY="your-api-key"

python main.py --input_file <输入.jsonl> --output_dir <输出目录>
```

**参数**：
- `--input_file`（必需）：输入 JSONL 路径，每行需含 `idx` 和 `problem` 字段
- `--output_dir`（必需）：每题结果 JSON 的输出目录

### 3.4 MCP Server（可选）

正常工作时，`PythonMCPClient` 直接 in-process 调用 `execute_python`（子进程隔离），无需独立启动 MCP server。如需作为标准 MCP server 独立运行：

```bash
python -m mcp_servers.python_executor.server
```

---

## 四、输出保存位置

### 4.1 结果文件

每题输出到 **`{output_dir}/{idx}.json`**，结构如下：

```json
{
  "idx": 0,
  "status": "success",
  "final_response": "最终答案：4/3",
  "trace": [
    {"step": "classification", "category": "数学分析", "confidence": 0.9, "candidates": ["数学分析", "测度积分"]},
    {"step": "reasoning", "attempts": 1, "answer": "4/3", "steps_count": 3},
    {"step": "python_verification", "attempts": 1, "success": true, "answer": "4/3"},
    {"step": "validation", "status": "match", "validated_answer": "4/3"},
    {"step": "coordination", "content": "<coordinator 生成的完整解题说明>", "response_length": 8}
  ]
}
```

- **`final_response` 格式**（遵循赛题"避免过长、答案明确"要求）：
  - 计算题 → `最终答案：{answer}`（仅简洁答案，完整解题过程记入 `trace` 的 `coordination.content`）
  - 证明题 → `结论：{answer}\n\n{证明过程}`（证明过程本身即答案主体，不可省略）
- **失败时**：`status: "error"`，含 `error.type` 与 `error.message`，`final_response` 为空串
- **断点续跑**：已存在且非空的输出文件会被跳过（`is_processed` 判断）

### 4.2 日志

日志输出到 **stderr（控制台）**。若安装了 `python-json-logger` 则为 JSON 格式，否则为纯文本。日志级别与目录由 `config.py` 的 `log_level`（默认 `INFO`）和 `log_dir`（默认 `logs`）配置。

### 4.3 配置

所有可调参数集中在 `config.py`：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `model` | `intern-s2-preview` | LLM 模型 |
| `max_retries_per_node` | 2 | 节点内重试次数 |
| `llm_max_retries` | 3 | LLM 调用重试次数 |
| `node_timeouts` | — | 各节点超时（秒） |
| `classifier_top_k` | 5 | 分类候选数 |
| `classifier_confidence_threshold` | 0.7 | 分类置信阈值 |
| `computation_tolerance` | 1e-6 | 计算题数值容差 |
| `reconciliation_max_rounds` | 2 | 调解重试上限 |
| `token_budget_max` | 30000 | Token 预算上限 |
| `temperatures` / `max_tokens` | — | 各节点 LLM 采样参数 |
