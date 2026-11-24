import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkBreaks from 'remark-breaks';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import {
  Activity,
  ArrowUp,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Clipboard,
  Copy,
  Eye,
  EyeOff,
  KeyRound,
  Menu,
  MessageSquarePlus,
  MoreHorizontal,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  Settings2,
  Square,
  Sparkles,
  Trash2,
  Trophy,
  X,
} from 'lucide-react';
import 'katex/dist/katex.min.css';
import './styles.css';

const API = import.meta.env.VITE_API_BASE || '';
// Quick Tunnels cannot reliably carry SSE; keep model streaming via short,
// cursor-based JSON reads. The query override also helps with other proxies.
const POLL_EVENTS = window.location.hostname.endsWith('.trycloudflare.com')
  || new URLSearchParams(window.location.search).get('transport') === 'poll';

const fallbackExamples = [
  { id: 'integral', difficulty: '基础', title: '定积分', plain_problem: '计算积分 ∫₀¹ x² dx。', check: '答案为 1/3' },
  { id: 'optimization', difficulty: '进阶', title: '二元函数最优化', plain_problem: '求 f(x,y)=x²+xy+y²−3x−3y 在实数范围内的最小值及取等点。', check: '最小值 −3，在 x=y=1 取得' },
  { id: 'number-theory', difficulty: '挑战', title: '数论证明', plain_problem: '证明：对每个整数 n>1，n⁴+4ⁿ 都是合数。', check: '证明覆盖奇偶两类' },
];

const stageMeta = {
  input: { label: '输入', hint: '题目已接收' },
  classifier: { label: '领域分类', hint: '识别题型与领域' },
  classification: { label: '领域分类', hint: '识别题型与领域' },
  database_retrieval: { label: '相似题检索', hint: '读取参考证据' },
  reasoning_agent: { label: '数学推理', hint: '构建解法' },
  reasoning: { label: '数学推理', hint: '构建解法' },
  python_agent: { label: 'Python 验证', hint: '独立复核' },
  python_verification: { label: 'Python 验证', hint: '独立复核' },
  objective_review: { label: '客观题复核', hint: '复核候选' },
  cross_validator: { label: '交叉验证', hint: '比较候选结果' },
  reconciliation: { label: '调解重试', hint: '处理分歧' },
  semantic_arbiter: { label: '语义仲裁', hint: '选择可信答案' },
  semantic_arbitration: { label: '语义仲裁', hint: '选择可信答案' },
  validation: { label: '答案验证', hint: '核对结论' },
  validation_history: { label: '验证记录', hint: '查看复核轮次' },
  problem_integrity: { label: '题面完整性', hint: '检查原始题面' },
  coordinator: { label: '答案整理', hint: '组织最终回答' },
  coordination: { label: '答案整理', hint: '组织最终回答' },
  timing: { label: '耗时统计', hint: '记录阶段耗时' },
  explanation: { label: '步骤解释', hint: '解释上一轮结果' },
  card_convention_gate: { label: '结果校准', hint: '应用确定性校准' },
  error: { label: '运行错误', hint: '阶段未正常完成' },
};

const traceFieldLabels = {
  purpose: '目的',
  status: '状态',
  category: '领域',
  category_path: '领域目录',
  selected_path: '选中目录',
  skill_document: '技能文档',
  validation_script: '验证脚本',
  classification_basis: '分类依据',
  question_mode: '题型模式',
  confidence: '置信度',
  signals: '识别信号',
  query: '检索问题',
  reference_count: '参考题数量',
  references: '参考题',
  results: '检索结果',
  rank: '排名',
  similarity: '相似度',
  source: '来源',
  problem: '题目',
  solution: '解法',
  analysis: '分析',
  steps: '推理过程',
  derivation: '推导过程',
  result: '结果',
  final_answer: '阶段结论',
  validation_points: '验证要点',
  answer_source: '答案来源',
  raw_response_excerpt: '模型响应节选',
  code_summary: '验证代码',
  code: '执行代码',
  python_code: 'Python 代码',
  expected_output_marker: '输出标记',
  extracted_answer: '提取答案',
  stdout: '标准输出',
  stderr: '错误输出',
  verification_trusted: '验证可信',
  authenticity_reasons: '可信度判断',
  evidence_status: '证据状态',
  evidence_summary: '证据摘要',
  contradictions: '矛盾记录',
  comparison: '比较结果',
  decision: '阶段决定',
  format: '输出格式',
  step_num: '步骤',
  description: '描述',
};

const codeFields = new Set([
  'code_summary',
  'code',
  'python_code',
  'stdout',
  'stderr',
  'raw_response_excerpt',
  'script',
]);

function traceKey(item) {
  return item?.step || item?.node || item?.type || 'event';
}

function traceMeta(itemOrKey) {
  const key = typeof itemOrKey === 'string' ? itemOrKey : traceKey(itemOrKey);
  return stageMeta[key] || { label: key.replaceAll('_', ' '), hint: '阶段事件' };
}

function sameStage(left, right) {
  const aliases = {
    classifier: 'classification',
    classification: 'classification',
    reasoning_agent: 'reasoning',
    reasoning: 'reasoning',
    python_agent: 'python_verification',
    python_verification: 'python_verification',
    semantic_arbiter: 'semantic_arbitration',
    semantic_arbitration: 'semantic_arbitration',
    coordinator: 'coordination',
    coordination: 'coordination',
  };
  return (aliases[left] || left) === (aliases[right] || right);
}

function isPresent(value) {
  return value !== undefined && value !== null && value !== '' && !(Array.isArray(value) && value.length === 0);
}

function fieldLabel(field) {
  return traceFieldLabels[field] || field.replaceAll('_', ' ');
}

function isCodeField(field) {
  return codeFields.has(field);
}

function traceSummary(item) {
  const value = item?.content || item?.payload?.summary || item?.thinking?.purpose || item?.thinking?.analysis || '阶段已完成。';
  const text = typeof value === 'string' ? value : JSON.stringify(value);
  return text.length > 220 ? `${text.slice(0, 217).trim()}…` : text;
}

function uid() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

// Mirrors run_service._conversation_title: a LaTeX problem is titled by the
// prose before its first formula ("计算二重积分") instead of raw markup.
function conversationTitle(message) {
  const text = message.replace(/\s+/g, ' ').trim();
  const lead = text.split('$')[0].replace(/[\s，。：:；;、,.]+$/, '');
  return (text.includes('$') && lead ? lead : text).slice(0, 28).trim() || '新聊天';
}

async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  });
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `请求失败 (${response.status})`);
  return response.json();
}

// The solver emits display math inline ("…$$formula$$\n下一步…").  remark-math
// only recognises a $$ block whose fences stand alone, so without this it
// swallows the following text into one broken math span (KaTeX renders it in
// red).  Fenced code is left untouched.
function normalizeMath(text) {
  return String(text || '')
    .split(/(```[\s\S]*?```)/g)
    .map((part, index) => (index % 2
      ? part
      : part.replace(/\$\$([\s\S]*?)\$\$/g, (_, body) => `\n\n$$\n${body.trim()}\n$$\n\n`)))
    .join('');
}

function Markdown({ children }) {
  return <ReactMarkdown remarkPlugins={[remarkMath, remarkBreaks, remarkGfm]} rehypePlugins={[rehypeKatex]}>{normalizeMath(children)}</ReactMarkdown>;
}

function Logo() {
  return <img className="brand-logo" src="/assets/sail-logo.png" alt="上海人工智能实验室" />;
}

function KeyControl({ value, onChange }) {
  const [open, setOpen] = useState(false);
  const [visible, setVisible] = useState(false);
  const [draft, setDraft] = useState(value);
  const shown = value ? `${value.slice(0, 5)}…${value.slice(-4)}` : '未设置 Key';
  return (
    <div className="key-area">
      {open && <div className="key-popover">
        <div className="popover-heading"><div><strong>API Key</strong><span>仅用于本机真实求解</span></div><button className="icon-button quiet" onClick={() => setOpen(false)} aria-label="关闭"><X size={16} /></button></div>
        <div className="key-input-wrap"><KeyRound size={15} /><input type={visible ? 'text' : 'password'} value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="sk-..." autoFocus /><button className="icon-button quiet" onClick={() => setVisible((current) => !current)} aria-label={visible ? '隐藏 Key' : '显示 Key'}>{visible ? <EyeOff size={15} /> : <Eye size={15} />}</button></div>
        <div className="key-actions"><button className="text-button" onClick={() => setDraft('')}>清除</button><button className="primary-small" onClick={() => { onChange(draft.trim()); setOpen(false); }}>保存 Key</button></div>
      </div>}
      <button className="key-button" onClick={() => { setDraft(value); setOpen((current) => !current); }}><span className={`key-dot ${value ? 'active' : ''}`} /><span className="key-copy"><b>Key</b><small>{shown}</small></span><MoreHorizontal size={16} /></button>
    </div>
  );
}

function TraceValue({ value, field = '', depth = 0 }) {
  if (!isPresent(value)) return null;
  if (depth > 4) {
    return <pre className="trace-json"><code>{JSON.stringify(value, null, 2)}</code></pre>;
  }
  if (typeof value === 'string') {
    return isCodeField(field)
      ? <pre className="trace-code"><code>{value}</code></pre>
      : <div className="trace-markdown"><Markdown>{value}</Markdown></div>;
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return <span className="trace-scalar">{String(value)}</span>;
  }
  if (Array.isArray(value)) {
    if (value.every((entry) => typeof entry === 'string' || typeof entry === 'number' || typeof entry === 'boolean')) {
      return <ul className="trace-bullets">{value.map((entry, index) => <li key={`${field}-${index}`}><TraceValue value={entry} field={field} depth={depth + 1} /></li>)}</ul>;
    }
    return <div className="trace-array">{value.map((entry, index) => <div className="trace-array-item" key={`${field}-${index}`}><span className="trace-array-index">{index + 1}</span><TraceValue value={entry} field={field} depth={depth + 1} /></div>)}</div>;
  }
  if (typeof value === 'object') {
    return <div className="trace-fields">{Object.entries(value).filter(([, entry]) => isPresent(entry)).map(([key, entry]) => <div className="trace-field" key={key}><span className="trace-field-label">{fieldLabel(key)}</span><div className="trace-field-value"><TraceValue value={entry} field={key} depth={depth + 1} /></div></div>)}</div>;
  }
  return null;
}

function ToolCall({ tool, index }) {
  if (!tool || typeof tool !== 'object') {
    return <div className="tool-call"><TraceValue value={tool} /></div>;
  }
  const { name, ...details } = tool;
  return <details className="tool-call" open={index === 0}>
    <summary><span className="tool-call-icon"><Settings2 size={13} /></span><strong>{name || '工具调用'}</strong><ChevronDown size={14} /></summary>
    <div className="tool-call-body"><TraceValue value={details} depth={1} /></div>
  </details>;
}

function TraceDetails({ item }) {
  const thinking = item?.thinking;
  const tools = Array.isArray(item?.tools) ? item.tools : [];
  const hasDetails = isPresent(thinking) || tools.length > 0;
  return <div className="trace-detail">
    {hasDetails && thinking && <section className="trace-detail-section"><div className="trace-detail-heading"><Sparkles size={13} /><span>思考内容</span></div><TraceValue value={thinking} depth={0} /></section>}
    {tools.length > 0 && <section className="trace-detail-section"><div className="trace-detail-heading"><Settings2 size={13} /><span>调用工具 <small>{tools.length}</small></span></div><div className="tool-list">{tools.map((tool, index) => <ToolCall key={`${tool?.name || 'tool'}-${index}`} tool={tool} index={index} />)}</div></section>}
    {!hasDetails && <div className="trace-detail-empty">这个阶段没有额外的结构化详情。</div>}
  </div>;
}

function NodeGraph({ events, trace, running, selectedNode, onSelect }) {
  const [expanded, setExpanded] = useState(false);
  const statuses = useMemo(() => {
    const map = {};
    for (const event of events) {
      if (!event.node) continue;
      // node.started -> "running": the spinner and highlight are keyed on it.
      const kind = event.type.replace('node.', '');
      map[event.node] = kind === 'started' ? 'running' : kind;
    }
    for (const item of trace || []) {
      if (item.status === 'cancelled') map[item.step] = 'cancelled';
    }
    return map;
  }, [events, trace]);
  const rows = [['input'], ['classifier'], ['database_retrieval'], ['reasoning_agent', 'python_agent'], ['cross_validator'], ['reconciliation', 'semantic_arbiter'], ['coordinator']];
  return <section className={`graph-panel ${expanded ? 'expanded' : ''}`}>
    <div className="panel-label"><span>运行轨迹</span><span className="panel-actions"><span className={`live-pill ${running ? 'on' : ''}`}><i />{running ? '实时' : '待运行'}</span><button className="graph-expand" onClick={() => setExpanded((current) => !current)} aria-label={expanded ? '收起完整拓扑' : '展开完整拓扑'}>{expanded ? <ChevronLeft size={13} /> : <ChevronRight size={13} />}</button></span></div>
    <div className="graph-canvas">
      <div className="graph-line" />
      {rows.map((row, rowIndex) => <div className={`graph-row row-${row.length}`} key={rowIndex}>
        {row.map((node) => {
          const status = statuses[node] || 'pending';
          const meta = stageMeta[node] || { label: node, hint: '' };
          return <button className={`node node-${status} ${selectedNode === node ? 'selected' : ''}`} key={node} onClick={() => onSelect(node)}>
            <span className="node-icon">{status === 'completed' ? <Check size={12} /> : status === 'running' ? <Activity size={12} /> : status === 'degraded' ? <CircleAlert size={12} /> : <span />}</span><span><b>{meta.label}</b><small>{meta.hint}</small></span>
          </button>;
        })}
      </div>)}
    </div>
    {selectedNode && <div className="node-detail"><strong>{traceMeta(selectedNode).label}</strong><span>{[...events].reverse().find((event) => event.node === selectedNode)?.payload?.summary || '该阶段尚未运行'}</span>{trace.find((item) => sameStage(traceKey(item), selectedNode)) && <TraceDetails item={trace.find((item) => sameStage(traceKey(item), selectedNode))} />}</div>}
  </section>;
}

function Sidebar({ collapsed, setCollapsed, conversation, conversations, groupedConversations, onNew, onSelect, onDelete, trace, events, running, mode, selectedNode, setSelectedNode, keyValue, setKeyValue }) {
  return <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
    <div className="sidebar-head"><Logo />{!collapsed && <button className="icon-button" onClick={() => setCollapsed(true)} aria-label="收起侧栏"><PanelLeftClose size={18} /></button>}{collapsed && <button className="icon-button" onClick={() => setCollapsed(false)} aria-label="展开侧栏"><PanelLeftOpen size={18} /></button>}</div>
    {!collapsed && <>
      <button className="new-chat" onClick={onNew}><MessageSquarePlus size={17} /><span>新聊天</span><span className="shortcut">⌘ K</span></button>
      {mode !== 'chat' && <NodeGraph trace={trace} events={events} running={running} selectedNode={selectedNode} onSelect={setSelectedNode} />}
      <div className="history">{conversations.length === 0 ? <><div className="section-caption">历史聊天</div><div className="history-empty">完成一次提问后会显示在这里</div></> : Object.entries(groupedConversations).map(([label, items]) => items.length > 0 && <div className="history-group" key={label}><div className="section-caption">{label}</div>{items.map((item) => <div className="history-item-wrap" key={item.id}><button className={`history-item ${conversation?.id === item.id ? 'current' : ''}`} onClick={() => onSelect(item.id)}><MessageSquarePlus size={14} /><span>{item.title || '新聊天'}</span></button><button className="history-delete" onClick={() => onDelete(item.id)} aria-label={`删除聊天 ${item.title || '新聊天'}`} title="删除聊天"><Trash2 size={14} /></button></div>)}</div>)}</div>
      <div className="sidebar-bottom"><KeyControl value={keyValue} onChange={setKeyValue} /></div>
    </>}
  </aside>;
}

function Composer({ onSubmit, onStop, disabled, running = false, stopping = false, thinking, onThinkingChange, compact = false }) {
  const [value, setValue] = useState('');
  const textarea = useRef(null);
  const submit = () => { const next = value.trim(); if (!next || disabled || running) return; onSubmit(next); setValue(''); };
  useEffect(() => { if (textarea.current) { textarea.current.style.height = 'auto'; textarea.current.style.height = `${Math.min(textarea.current.scrollHeight, 160)}px`; } }, [value]);
  return <div className={`composer ${compact ? 'compact' : ''}`}>
    <textarea ref={textarea} value={value} onChange={(event) => setValue(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); submit(); } }} placeholder="询问ICMA" rows={1} disabled={disabled || running} />
    <div className="composer-tools">
      <button className={`thinking-toggle ${thinking ? 'selected' : ''}`} aria-pressed={thinking} title={thinking ? '已开启：使用完整 ICMA 框架求解与验证' : '已关闭：直接对话，流式输出回答'} onClick={() => onThinkingChange(!thinking)}><Sparkles size={14} />完整智能体</button>
      <span className="tool-note">{thinking ? 'ICMA 思考模式' : '普通对话 · 流式输出'}</span>
      <button className="stop-button" disabled={!running || stopping || disabled} onClick={onStop} aria-label="停止生成" title={!running ? '生成开始后可停止' : disabled && !stopping ? '正在启动运行，请稍候' : '强制停止当前运行'}><Square size={13} />{stopping ? '正在停止' : '停止生成'}</button>
      {!running && <button className="send-button" disabled={!value.trim() || disabled} onClick={submit} aria-label="发送"><ArrowUp size={18} /></button>}
    </div>
  </div>;
}

function ExampleCard({ example, onClick, onLive }) {
  return <div className="example-card" onClick={onClick} role="button" tabIndex={0} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') onClick(); }}><div className="example-top"><span className="difficulty">{example.difficulty}</span><span className="example-arrow"><ArrowUp size={14} /></span></div><strong>{example.title}</strong><span className="example-problem">{example.plain_problem}</span><div className="example-foot"><small>{example.check}</small><button className="recompute-button" onClick={(event) => { event.stopPropagation(); onLive(); }}>真实重算</button></div></div>;
}

function CompetitionCard({ example, onClick }) {
  return <div className="example-card competition-card" onClick={onClick} role="button" tabIndex={0} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') onClick(); }}>
    <div className="example-top"><span className="difficulty">{example.difficulty}</span><span className="example-arrow"><ArrowUp size={14} /></span></div>
    <strong>{example.title}</strong>
    <div className="example-problem competition-problem"><Markdown>{example.problem}</Markdown></div>
    <div className="example-foot"><small>{example.source}</small></div>
  </div>;
}

function TracePanel({ trace, mode }) {
  const [open, setOpen] = useState(true);
  return <div className={`trace-panel ${open ? 'open' : ''}`}><button className="trace-toggle" onClick={() => setOpen((current) => !current)}><span><Sparkles size={15} />{mode === 'replay' ? '示例回放过程' : '已思考过程'}<small>{trace.length ? `${trace.length} 个阶段` : '阶段事件'}</small></span><ChevronDown size={17} /></button>{open && <div className="trace-list">{trace.length ? trace.map((item, index) => <TraceItem item={item} index={index} key={`${traceKey(item)}-${index}`} />) : <div className="trace-empty">等待阶段事件…</div>}</div>}</div>;
}

const AGENT_STAGES = new Set(['reasoning_agent', 'reasoning', 'python_agent', 'python_verification']);

// Re-renders once a second while a stage runs so its elapsed time ticks.
function useNow(active) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return undefined;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active]);
  return now;
}

function TraceItem({ item, index }) {
  const [expanded, setExpanded] = useState(false);
  const status = item?.status || 'completed';
  const meta = traceMeta(item);
  const startedAt = status === 'running' && item?.started_at ? Date.parse(item.started_at) : NaN;
  const ticking = Number.isFinite(startedAt);
  const now = useNow(ticking);
  const statusLabel = status === 'failed' ? '失败' : status === 'cancelled' ? '已停止' : status === 'running' ? (AGENT_STAGES.has(traceKey(item)) ? '思考中' : '运行中') : status === 'degraded' ? '降级完成' : '已完成';
  const seconds = ticking ? `${Math.max(0, Math.floor((now - startedAt) / 1000))}s` : item?.elapsed_s ? `${item.elapsed_s}s` : '';
  return <div className={`trace-item trace-item-${status} ${expanded ? 'expanded' : ''}`}>
    <button className="trace-item-toggle" aria-expanded={expanded} onClick={() => setExpanded((current) => !current)}>
      <span className={`trace-dot trace-dot-${status}`} />
      <span className="trace-item-copy"><span className="trace-item-title"><strong>{meta.label}</strong><small>{statusLabel}{seconds ? ` · ${seconds}` : ''}</small></span><span className="trace-item-summary">{traceSummary(item)}</span></span>
      <ChevronDown className="trace-item-chevron" size={15} />
    </button>
    {expanded && <TraceDetails item={item} />}
  </div>;
}

function mergeTraceEvent(trace, event) {
  if (event?.type === 'message.reasoning.delta') {
    const content = event.payload?.content;
    if (typeof content !== 'string' || !content) return trace;
    const index = trace.findIndex((item) => item.step === 'model_reasoning');
    const previous = index < 0 ? null : trace[index];
    const item = {
      step: 'model_reasoning', status: 'running', content: '模型正在思考',
      thinking: { analysis: `${previous?.thinking?.analysis || ''}${content}` },
    };
    if (index < 0) return [...trace, item];
    return trace.map((entry, current) => current === index ? item : entry);
  }
  if (event?.type === 'run.cancelled') {
    return trace.map((item) => item.status === 'running' ? {
      ...item,
      status: 'cancelled',
      content: item.step === 'model_reasoning' ? '模型思考已停止' : item.content || '运行在此阶段停止',
    } : item);
  }
  if (['message.delta', 'run.completed', 'run.failed'].includes(event?.type)) {
    return trace.map((item) => item.step === 'model_reasoning' ? {
      ...item,
      status: event.type === 'run.failed' ? 'failed' : 'completed',
      content: event.type === 'run.failed' ? '模型思考中断' : '模型思考完成',
    } : item);
  }
  if (!event?.node || !event.type?.startsWith('node.')) return trace;
  const payload = event.payload || {};
  const status = event.type === 'node.started'
    ? 'running'
    : event.type === 'node.degraded'
      ? 'degraded'
      : event.type === 'node.failed'
        ? 'failed'
        : 'completed';
  const item = {
    step: event.node,
    status,
    content: payload.summary || '',
    elapsed_s: payload.elapsed_s,
    thinking: payload.thinking || {},
    tools: payload.tools || [],
    // Server time of node.started anchors the ticking timer, so a refresh
    // mid-run resumes the count instead of restarting it from zero.
    ...(status === 'running' && event.timestamp ? { started_at: event.timestamp } : {}),
  };
  const existingIndex = [...trace].map((entry, index) => ({ entry, index }))
    .reverse()
    .find(({ entry }) => entry.step === event.node && entry.status === 'running')?.index;
  if (existingIndex === undefined) return [...trace, item];
  const next = [...trace];
  next[existingIndex] = { ...next[existingIndex], ...item };
  return next;
}

function traceFromEvents(events) {
  return (events || []).reduce((trace, event) => mergeTraceEvent(trace, event), []);
}

function ReasoningPanel({ trace, streaming, answered }) {
  const [open, setOpen] = useState(true);
  const item = (trace || []).find((entry) => entry.step === 'model_reasoning');
  const content = item?.thinking?.analysis || '';
  const failed = item?.status === 'failed';
  const cancelled = item?.status === 'cancelled';
  const thinking = streaming && !answered && !failed && !cancelled;
  if (!content && !thinking) return null;
  return <div className={`trace-panel reasoning-panel ${open ? 'open' : ''}`}>
    <button className="trace-toggle" aria-expanded={open} aria-label={open ? '收起模型思考过程' : '展开模型思考过程'} onClick={() => setOpen((current) => !current)}>
      <span><Sparkles size={15} />{cancelled ? '思考已停止' : failed ? '思考已中断' : thinking ? '正在思考' : '已思考'}<small>模型推理</small></span><ChevronDown size={17} />
    </button>
    {open && <div className="reasoning-body trace-markdown">{content ? <Markdown>{content}</Markdown> : <span className="reasoning-wait">等待模型返回思考内容…</span>}</div>}
  </div>;
}

function Message({ message }) {
  const isUser = message.role === 'user';
  const [copied, setCopied] = useState(false);
  const copy = async () => { await navigator.clipboard?.writeText(message.content); setCopied(true); setTimeout(() => setCopied(false), 1200); };
  return <article className={`message ${isUser ? 'user-message' : 'assistant-message'}`}>
    <div className="message-avatar">{isUser ? '你' : <Sparkles size={15} />}</div>
    <div className="message-body">
      <div className="message-meta"><strong>{isUser ? '你' : 'ICMA'}</strong>{!isUser && message.mode === 'replay' && <span className="replay-badge">示例回放</span>}{!isUser && message.mode !== 'replay' && <span className="live-badge">{message.mode === 'chat' ? '普通对话' : '实时求解'}</span>}{!isUser && message.run_status === 'cancelled' && <span className="cancelled-badge">已停止</span>}</div>
      {!isUser && message.mode === 'chat' && <ReasoningPanel trace={message.trace} streaming={message.streaming} answered={Boolean(message.content)} />}
      <div className="markdown"><Markdown>{message.content}</Markdown></div>
      {!isUser && <>{message.mode !== 'chat' && <TracePanel trace={message.trace || []} mode={message.mode} />}{!message.streaming && <div className="message-actions"><button onClick={copy}>{copied ? <Check size={13} /> : <Copy size={13} />}{copied ? '已复制' : '复制'}</button></div>}</>}
    </div>
  </article>;
}

function EmptyState({ examples, onExample, onLive, onCompetition, onSubmit, thinking, onThinkingChange }) {
  const samples = examples.filter((example) => example.group !== 'competition');
  const competition = examples.filter((example) => example.group === 'competition');
  return <div className="empty-state"><img src="/assets/icma-mark.svg" alt="ICMA" className="icma-mark" /><h1>把数学题交给 ICMA</h1><p>从可验证的推理开始，查看每一步如何抵达答案。</p><div className="home-composer"><Composer onSubmit={onSubmit} thinking={thinking} onThinkingChange={onThinkingChange} /></div><div className="example-grid">{samples.map((example) => <ExampleCard key={example.id} example={example} onClick={() => onExample(example)} onLive={() => onLive(example)} />)}</div>
    {competition.length > 0 && <section className="competition-section" aria-label="试试竞赛题"><div className="competition-heading"><Trophy size={15} /><span>试试竞赛题</span></div><div className="competition-grid">{competition.map((example) => <CompetitionCard key={example.id} example={example} onClick={() => onCompetition(example)} />)}</div></section>}
  </div>;
}

function App() {
  const [collapsed, setCollapsed] = useState(false);
  const [conversation, setConversation] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [examples, setExamples] = useState(fallbackExamples);
  const [selectedNode, setSelectedNode] = useState(null);
  const [runStates, setRunStates] = useState({});
  const [keyValue, setKeyValue] = useState(() => localStorage.getItem('icma-demo-key') ?? '');
  const [thinking, setThinking] = useState(() => localStorage.getItem('icma-demo-thinking') !== 'false');
  const [mobileOpen, setMobileOpen] = useState(false);
  const [error, setError] = useState('');
  const runStatesRef = useRef({});
  const resourcesRef = useRef(new Map());
  const finalizedRunsRef = useRef(new Set());
  const activeConversationRef = useRef(null);
  const selectionRef = useRef(0);
  const followOutputRef = useRef(true);

  const updateRunState = (conversationId, updater) => {
    setRunStates((current) => {
      const previous = current[conversationId] || { events: [], trace: [], running: false };
      const nextState = typeof updater === 'function' ? updater(previous) : { ...previous, ...updater };
      const next = { ...current, [conversationId]: nextState };
      runStatesRef.current = next;
      return next;
    });
  };

  const closeRunResource = (runId) => {
    const resource = resourcesRef.current.get(runId);
    if (!resource) return;
    if (resource.source) resource.source.close();
    if (resource.reconnectTimer) clearTimeout(resource.reconnectTimer);
    if (resource.pollTimer) clearTimeout(resource.pollTimer);
    resourcesRef.current.delete(runId);
  };

  useEffect(() => {
    activeConversationRef.current = conversation?.id || null;
  }, [conversation?.id]);
  useEffect(() => { localStorage.setItem('icma-demo-key', keyValue); }, [keyValue]);
  useEffect(() => { localStorage.setItem('icma-demo-thinking', String(thinking)); }, [thinking]);
  useEffect(() => {
    let lastScrollY = window.scrollY;
    const trackScroll = () => {
      const currentY = window.scrollY;
      if (currentY < lastScrollY - 2) followOutputRef.current = false;
      else if (document.documentElement.scrollHeight - currentY - window.innerHeight < 40) followOutputRef.current = true;
      lastScrollY = currentY;
    };
    window.addEventListener('scroll', trackScroll, { passive: true });
    return () => window.removeEventListener('scroll', trackScroll);
  }, []);
  useEffect(() => () => {
    resourcesRef.current.forEach((resource) => {
      resource.source?.close();
      if (resource.reconnectTimer) clearTimeout(resource.reconnectTimer);
      if (resource.pollTimer) clearTimeout(resource.pollTimer);
    });
  }, []);

  const loadRunEvents = async (runId) => {
    if (!runId) return [];
    const events = [];
    let after = 0;
    while (true) {
      const batch = await api(`/api/runs/${runId}/event-batch?after=${after}`);
      events.push(...batch.events);
      if (!batch.has_more || !batch.events.length) return events;
      after = Number(batch.events.at(-1).seq);
    }
  };

  const finishRun = async (conversationId, runId, status) => {
    if (finalizedRunsRef.current.has(runId)) return;
    finalizedRunsRef.current.add(runId);
    closeRunResource(runId);
    let nextConversation = null;
    try {
      nextConversation = await api(`/api/conversations/${conversationId}`);
      setConversations(await api('/api/conversations'));
    } catch (err) {
      if (activeConversationRef.current === conversationId) setError(err.message);
    }
    updateRunState(conversationId, (previous) => {
      if (previous.runId !== runId) return previous;
      const assistant = nextConversation?.messages?.find((message) => message.role === 'assistant' && message.run_id === runId);
      const pending = assistant ? null : previous.pending ? {
        ...previous.pending,
        streaming: false,
        run_status: status,
        content: status === 'cancelled' ? (previous.mode === 'chat' && previous.pending.content) || '生成已停止。' : status === 'failed' ? (previous.mode === 'chat' && previous.pending.content) || '本次运行失败，请查看错误提示后重试。' : previous.pending.content,
      } : null;
      return {
        ...previous,
        running: false,
        starting: false,
        stopping: false,
        pending,
        trace: assistant ? assistant.trace || [] : status === 'cancelled' ? mergeTraceEvent(previous.trace, { type: 'run.cancelled' }) : previous.trace,
      };
    });
    if (activeConversationRef.current === conversationId && nextConversation) {
      setConversation(nextConversation);
      setSelectedNode(null);
      if (status === 'failed') setError((await api(`/api/runs/${runId}`).catch(() => ({}))).error || '运行失败');
    }
  };

  const connectRun = (conversationId, runId, after = 0) => {
    if (!runId || finalizedRunsRef.current.has(runId) || resourcesRef.current.has(runId)) return;
    const resource = { source: null, reconnectTimer: null, pollTimer: null, lastSeq: after, polling: POLL_EVENTS };
    resourcesRef.current.set(runId, resource);
    const isActive = () => resourcesRef.current.get(runId) === resource && !finalizedRunsRef.current.has(runId);
    const receive = (next) => {
      if (next.run_id !== runId || !isActive()) return;
      if (Number(next.seq) <= resource.lastSeq) return;
      resource.lastSeq = Math.max(resource.lastSeq, Number(next.seq) || 0);
      updateRunState(conversationId, (previous) => {
        if (previous.events.some((item) => item.seq === next.seq)) return previous;
        const events = [...previous.events, next];
        const mode = next.type === 'run.started' ? next.payload?.mode || previous.mode : previous.mode;
        const pending = next.type === 'message.delta' ? {
          ...(previous.pending || { id: `pending-${runId}`, role: 'assistant', mode: 'chat', trace: [] }),
          content: `${previous.pending?.content || ''}${next.payload?.content || ''}`,
          streaming: true,
        } : previous.pending;
        return {
          ...previous,
          runId,
          events,
          mode,
          pending,
          trace: mergeTraceEvent(previous.trace, next),
          running: true,
        };
      });
    };
    const open = () => {
      if (!isActive() || resource.polling) return;
      const source = new EventSource(`${API}/api/runs/${runId}/events?after=${resource.lastSeq}`);
      resource.source = source;
      source.onmessage = (event) => {
        try {
          receive(JSON.parse(event.data));
        } catch {
          // A malformed event should not terminate the run subscription.
        }
      };
      source.onerror = () => {
        source.close();
        resource.source = null;
        // Resume after the last received sequence without duplicating tokens.
        resource.polling = true;
      };
    };
    open();
    const poll = async () => {
      if (!isActive()) return;
      let delay = resource.polling ? 400 : 700;
      try {
        const current = resource.polling
          ? await api(`/api/runs/${runId}/event-batch?after=${resource.lastSeq}`)
          : await api(`/api/runs/${runId}`);
        if (!isActive()) return;
        if (current.events) current.events.forEach(receive);
        if (['completed', 'failed', 'cancelled'].includes(current.status) && !current.has_more) {
          await finishRun(conversationId, runId, current.status);
          return;
        }
        if (current.has_more) delay = 0;
      } catch {
        // Retry transient failures using the last successfully received cursor.
        delay = 1500;
      }
      if (isActive()) resource.pollTimer = setTimeout(poll, delay);
    };
    resource.pollTimer = setTimeout(poll, 700);
  };

  const hydrateConversation = async (id, token = selectionRef.current) => {
    const next = await api(`/api/conversations/${id}`);
    const latestRunId = [...(next.messages || [])].reverse().find((message) => message.run_id)?.run_id;
    const previous = runStatesRef.current[id] || { events: [], trace: [], running: false };
    let nextState = { ...previous, runId: latestRunId || previous.runId };
    if (latestRunId) {
      const info = await api(`/api/runs/${latestRunId}`).catch(() => null);
      nextState.mode = info?.mode || nextState.mode;
      if (info && ['completed', 'failed', 'cancelled'].includes(info.status)) {
        const loadedEvents = await loadRunEvents(latestRunId);
        const assistant = (next.messages || []).find((message) => message.role === 'assistant' && message.run_id === latestRunId);
        nextState = {
          ...nextState,
          events: loadedEvents,
          trace: assistant?.trace?.length ? assistant.trace : traceFromEvents(loadedEvents),
          running: false,
          starting: false,
          pending: info.status === 'failed' && !assistant ? {
            id: `pending-${latestRunId}`, role: 'assistant', mode: info.mode, trace: [],
            content: '本次运行失败，请查看错误提示后重试。',
          } : null,
        };
        if (info.status === 'failed' && token === selectionRef.current) setError(info.error || '运行失败');
      } else {
        nextState = {
          ...nextState,
          running: true,
          starting: false,
          pending: previous.pending || {
            id: `pending-${latestRunId}`,
            role: 'assistant',
            content: info?.mode === 'chat' ? '' : '正在逐阶段整理结果…',
            mode: info?.mode || 'live',
            trace: [],
            streaming: true,
          },
        };
      }
    } else {
      nextState = { ...nextState, events: [], trace: [], running: false, starting: false, pending: null };
    }
    updateRunState(id, nextState);
    if (latestRunId && nextState.running) {
      connectRun(id, latestRunId, nextState.events.reduce((max, item) => Math.max(max, Number(item.seq) || 0), 0));
    }
    if (token === selectionRef.current) {
      setConversation(next);
      setSelectedNode(null);
    }
    return next;
  };

  useEffect(() => {
    Promise.all([
      api('/api/examples').catch(() => fallbackExamples),
      api('/api/conversations').catch(() => []),
    ]).then(async ([loadedExamples, loadedConversations]) => {
      setExamples(loadedExamples);
      setConversations(loadedConversations);
      const latest = loadedConversations[0];
      if (latest) await hydrateConversation(latest.id).catch(() => {});
    });
  }, []);
  const newConversation = async () => {
    setError('');
    if (conversation && !(conversation.messages || []).length && conversation.title === '新聊天') {
      setSelectedNode(null);
      setMobileOpen(false);
      return;
    }
    try {
      const next = await api('/api/conversations', { method: 'POST' });
      setConversation({ ...next, messages: [] });
      setConversations((current) => [next, ...current.filter((item) => item.id !== next.id)]);
      updateRunState(next.id, { events: [], trace: [], running: false, pending: null });
      setSelectedNode(null);
      setMobileOpen(false);
    } catch (err) {
      setError(err.message);
    }
  };
  const chooseConversation = async (id) => {
    const token = ++selectionRef.current;
    try {
      await hydrateConversation(id, token);
      setMobileOpen(false);
    } catch (err) {
      if (token === selectionRef.current) setError(err.message);
    }
  };
  const deleteConversation = async (id) => {
    if (runStatesRef.current[id]?.running) {
      setError('该聊天正在运行，完成后才能删除。');
      return;
    }
    const target = conversations.find((item) => item.id === id);
    const title = target?.title || '新聊天';
    if (!window.confirm(`删除“${title}”？聊天记录和运行轨迹都会被删除。`)) return;
    try {
      await api(`/api/conversations/${id}`, { method: 'DELETE' });
      const remaining = conversations.filter((item) => item.id !== id);
      setConversations(remaining);
      if (conversation?.id !== id) return;
      const next = remaining[0];
      if (next) {
        await chooseConversation(next.id);
      } else {
        setConversation(null);
        setSelectedNode(null);
      }
    } catch (err) {
      setError(err.message);
    }
  };
  const start = async (message, mode = thinking ? 'live' : 'chat', exampleId = null) => {
    setError('');
    let active = conversation;
    if (!active) {
      setError('请先点击左侧“新聊天”创建聊天窗口，再开始提问。');
      return;
    }
    if (runStatesRef.current[active.id]?.running) {
      setError('当前聊天正在运行，请先切换聊天或等待它完成。');
      return;
    }
    followOutputRef.current = true;
    const localUser = { id: uid(), role: 'user', content: message, mode, trace: [] };
    const nextTitle = active.title === '新聊天' ? conversationTitle(message) : active.title;
    setConversation((current) => ({ ...current, title: nextTitle, messages: [...(current?.messages || []), localUser] }));
    setConversations((current) => current.map((item) => item.id === active.id
      ? { ...item, title: nextTitle, message_count: Math.max(Number(item.message_count || 0), 1), updated_at: new Date().toISOString() }
      : item));
    updateRunState(active.id, { events: [], trace: [], mode, running: true, starting: true, stopping: false, pending: null });
    setSelectedNode(null);
    try {
      const run = await api(`/api/conversations/${active.id}/runs`, { method: 'POST', headers: { 'X-ICMA-Key': keyValue }, body: JSON.stringify({ message, mode, example_id: exampleId, request_id: uid() }) });
      updateRunState(active.id, {
        runId: run.id,
        mode: run.mode,
        running: true,
        starting: false,
        stopping: false,
        pending: { id: `pending-${run.id}`, role: 'assistant', content: run.mode === 'chat' ? '' : '正在逐阶段整理结果…', mode: run.mode, trace: [], streaming: true },
      });
      connectRun(active.id, run.id);
    } catch (err) {
      updateRunState(active.id, (previous) => ({ ...previous, running: false, starting: false, pending: { id: `failed-${uid()}`, role: 'assistant', mode, content: '本次运行未能启动。', trace: previous.trace } }));
      if (activeConversationRef.current === active.id) setError(err.message);
    }
  };
  const stop = async () => {
    const active = conversation;
    const state = active && runStatesRef.current[active.id];
    if (!active || !state?.runId || !state.running || state.stopping) return;
    setError('');
    updateRunState(active.id, { stopping: true });
    try {
      const result = await api(`/api/runs/${state.runId}/cancel`, { method: 'POST' });
      if (['completed', 'failed', 'cancelled'].includes(result.status)) {
        await finishRun(active.id, state.runId, result.status);
      }
    } catch (err) {
      updateRunState(active.id, { stopping: false });
      if (activeConversationRef.current === active.id) setError(err.message);
    }
  };
  const isEmpty = !conversation?.messages?.length;
  const visibleConversations = useMemo(() => conversations.filter((item) => item.id === conversation?.id || Number(item.message_count || 0) > 0), [conversations, conversation?.id]);
  const currentRunState = conversation ? runStates[conversation.id] : null;
  const events = currentRunState?.events || [];
  const running = Boolean(currentRunState?.running);
  const latestTrace = useMemo(() => currentRunState
    ? currentRunState.trace || []
    : [...(conversation?.messages || [])].reverse().find((message) => message.role === 'assistant')?.trace || [], [currentRunState, conversation]);
  const displayMessages = useMemo(() => {
    const base = conversation?.messages || [];
    const pending = currentRunState?.pending;
    return pending && !base.some((message) => message.id === pending.id) ? [...base, { ...pending, trace: currentRunState.trace || [] }] : base;
  }, [conversation, currentRunState]);
  useEffect(() => { followOutputRef.current = true; }, [conversation?.id]);
  useEffect(() => {
    if (!conversation?.id || !followOutputRef.current) return undefined;
    const frame = requestAnimationFrame(() => {
      if (followOutputRef.current) window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'auto' });
    });
    return () => cancelAnimationFrame(frame);
  }, [conversation?.id, currentRunState?.running, displayMessages]);
  const groupedConversations = useMemo(() => {
    const groups = { 今天: [], 昨天: [], 更早: [] };
    const today = new Date().toDateString();
    const yesterday = new Date(Date.now() - 86400000).toDateString();
    visibleConversations.forEach((item) => { const date = new Date(item.updated_at).toDateString(); groups[date === today ? '今天' : date === yesterday ? '昨天' : '更早'].push(item); });
    return groups;
  }, [visibleConversations]);
  return <div className="app-shell">
    <div className={`mobile-scrim ${mobileOpen ? 'visible' : ''}`} onClick={() => setMobileOpen(false)} />
    <div className={`sidebar-wrap ${collapsed ? 'collapsed' : ''} ${mobileOpen ? 'mobile-visible' : ''}`}><Sidebar collapsed={collapsed} setCollapsed={setCollapsed} conversation={conversation} conversations={visibleConversations} groupedConversations={groupedConversations} onNew={newConversation} onSelect={chooseConversation} onDelete={deleteConversation} trace={latestTrace} events={events} running={running} mode={currentRunState?.mode} selectedNode={selectedNode} setSelectedNode={setSelectedNode} keyValue={keyValue} setKeyValue={setKeyValue} /></div>
    <main className="main-panel"><header className="mobile-header"><button className="icon-button" onClick={() => setMobileOpen(true)} aria-label="打开侧栏"><Menu size={19} /></button><span>ICMA</span><button className="icon-button" onClick={newConversation} aria-label="新聊天"><MessageSquarePlus size={18} /></button></header>
      {error && <div className="error-banner"><CircleAlert size={16} />{error}<button onClick={() => setError('')}><X size={15} /></button></div>}
      {isEmpty ? <EmptyState examples={examples} onExample={(example) => start(example.plain_problem, 'replay', example.id)} onLive={(example) => start(example.plain_problem)} onCompetition={(example) => start(example.problem)} onSubmit={(message) => start(message)} thinking={thinking} onThinkingChange={setThinking} /> : <div className="conversation-view"><div className="conversation-head"><span>数学推理工作台</span>{running && <span className="working"><i />{currentRunState?.mode === 'chat' ? '正在回复' : '正在运行'}</span>}</div><div className="messages">{displayMessages.map((message) => <Message key={message.id} message={message} />)}{running && <div className="typing"><span className="typing-dot" /><span>{currentRunState?.mode === 'chat' ? 'ICMA 正在回复…' : 'ICMA 正在整理阶段结果'}</span></div>}</div><div className="bottom-composer"><Composer onSubmit={(message) => start(message)} onStop={stop} disabled={Boolean(currentRunState?.starting) || Boolean(currentRunState?.stopping)} running={running} stopping={Boolean(currentRunState?.stopping)} thinking={thinking} onThinkingChange={setThinking} compact /></div></div>}
    </main>
  </div>;
}

createRoot(document.getElementById('root')).render(<App />);
