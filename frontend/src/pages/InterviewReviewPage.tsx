import { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:8000';

// ── Types ─────────────────────────────────────────────────────────────────────

interface ResultMeta {
  filename:   string;
  saved_at:   string;
  direction:  string;
  jd_snippet: string;
  sm_state:   string;
  task_count: number;
  avg_score:  number | null;
}

interface ThoughtNode {
  id:              string;
  node_type:       'task' | 'question';
  text:            string;
  answer:          string;
  depth:           number;
  status:          string;
  score:           number | null;
  verdict:         string | null;
  feedback:        string;
  reasoning:       string;
  director_note:   string;
  question_intent: string;
  task_type:       string;
  summary:         string;
  children:        ThoughtNode[];
}

const verdictLabel = (v: string) =>
  v === 'deepen' ? '深挖' : v === 'pivot' ? '转向' : v === 'back_up' ? '退层' : v;

interface SessionResult {
  session_id: string;
  saved_at:   string;
  direction:  string;
  jd:         string;
  sm_final:   { state: string };
  tree:       ThoughtNode[];
  history:    { role: string; content: string }[];
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtDate(saved_at: string): string {
  // saved_at: "20260321_143000"
  if (!saved_at || saved_at.length < 15) return saved_at;
  const y = saved_at.slice(0, 4);
  const mo = saved_at.slice(4, 6);
  const d = saved_at.slice(6, 8);
  const h = saved_at.slice(9, 11);
  const mi = saved_at.slice(11, 13);
  return `${y}-${mo}-${d} ${h}:${mi}`;
}

// ── Tree view (read-only, collapsible) ────────────────────────────────────────

function TreeView({ roots }: { roots: ThoughtNode[] }) {
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const toggle = (id: string) =>
    setCollapsed(prev => { const s = new Set(prev); s.has(id) ? s.delete(id) : s.add(id); return s; });

  const renderNode = (node: ThoughtNode, indent = 0): React.ReactNode => {
    const isTask = node.node_type === 'task';
    const hasChildren = node.children.length > 0;
    const isCollapsed = collapsed.has(node.id);
    return (
      <div key={node.id}
        className={`rv-node rv-node--${node.node_type} rv-node--${node.status}`}
        style={{ marginLeft: indent * 16 }}
      >
        <div className="rv-node-row"
          onClick={() => hasChildren && toggle(node.id)}
          style={{ cursor: hasChildren ? 'pointer' : 'default' }}
        >
          <span className="rv-node-collapse">
            {hasChildren ? (isCollapsed ? '▸' : '▾') : ''}
          </span>
          <span className="rv-node-bullet">
            {isTask
              ? (node.status === 'done' ? '✓' : '○')
              : (node.score !== null ? (node.score >= 4 ? '✓' : '·') : '·')}
          </span>
          <span className={`rv-node-text ${isTask ? 'rv-node-text--task' : 'rv-node-text--question'}`}>
            {node.text}
          </span>
          {node.score !== null && (
            <span className={`rv-score rv-score--${node.score >= 4 ? 'good' : node.score <= 2 ? 'low' : 'mid'}`}>
              {node.score}/5
            </span>
          )}
          {node.verdict && node.verdict !== 'pass' && (
            <span className={`rv-verdict rv-verdict--${node.verdict}`}>
              🎬 {verdictLabel(node.verdict)}
            </span>
          )}
        </div>
        {/* 面试官：出题意图 */}
        {node.question_intent && !isTask && node.status !== 'planned' && node.status !== 'skipped' && (
          <div className="rv-node-attr rv-node-attr--interviewer">
            <span className="rv-attr-label">面试官</span> {node.question_intent}
          </div>
        )}
        {/* 待问 / 已跳过 */}
        {(node.status === 'planned' || node.status === 'skipped') && (
          <div className={`rv-node-attr rv-node-attr--${node.status}`}>
            <span className="rv-attr-label">{node.status === 'skipped' ? '已跳过' : '待问'}</span>
            {node.question_intent}
          </div>
        )}
        {/* 候选人回答 */}
        {node.answer && !isTask && (
          <div className="rv-node-answer">
            <span className="rv-attr-label">候选人</span> {node.answer}
          </div>
        )}
        {/* 评分员：CoT 分析 */}
        {node.reasoning && (
          <div className="rv-node-attr rv-node-attr--scorer">
            <span className="rv-attr-label">评分员</span> {node.reasoning}
          </div>
        )}
        {/* 评分员：简要反馈 */}
        {node.feedback && (
          <div className="rv-node-attr rv-node-attr--feedback">
            <span className="rv-attr-label">反馈</span> {node.feedback}
          </div>
        )}
        {/* 导演：决策理由 */}
        {node.director_note && node.verdict && (
          <div className="rv-node-attr rv-node-attr--director">
            <span className="rv-attr-label">导演 {verdictLabel(node.verdict)}</span> {node.director_note}
          </div>
        )}
        {!isCollapsed && node.children.map(c => renderNode(c, indent + 1))}
      </div>
    );
  };

  if (roots.length === 0) return <p className="rv-empty">暂无数据</p>;
  return <div className="rv-tree">{roots.map(n => renderNode(n))}</div>;
}

// ── Session detail ────────────────────────────────────────────────────────────

function SessionDetail({ filename }: { filename: string }) {
  const [data, setData] = useState<SessionResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<'tree' | 'chat'>('tree');

  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/interview/results/${filename}`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [filename]);

  if (loading) return <div className="rv-detail-empty">加载中…</div>;
  if (!data)   return <div className="rv-detail-empty">加载失败</div>;

  const scores = data.tree
    .flatMap(n => flatNodes(n))
    .filter(n => n.score !== null)
    .map(n => n.score as number);
  const avg = scores.length ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1) : '—';

  return (
    <div className="rv-detail">
      {/* Header */}
      <div className="rv-detail-header">
        <div className="rv-detail-meta">
          <span className="rv-detail-date">{fmtDate(data.saved_at)}</span>
          {data.direction && <span className="rv-detail-dir">{data.direction}</span>}
          <span className={`rv-detail-state rv-detail-state--${data.sm_final?.state === 'DONE' ? 'done' : 'mid'}`}>
            {data.sm_final?.state}
          </span>
          <span className="rv-detail-score">平均分 {avg}</span>
        </div>
        {data.jd && <p className="rv-detail-jd">{data.jd.slice(0, 120)}{data.jd.length > 120 ? '…' : ''}</p>}
      </div>

      {/* Tabs */}
      <div className="rv-tabs">
        <button className={`rv-tab ${tab === 'tree' ? 'rv-tab--active' : ''}`} onClick={() => setTab('tree')}>
          思维树
        </button>
        <button className={`rv-tab ${tab === 'chat' ? 'rv-tab--active' : ''}`} onClick={() => setTab('chat')}>
          对话记录
        </button>
      </div>

      {/* Content */}
      <div className="rv-detail-body">
        {tab === 'tree' ? (
          <TreeView roots={data.tree} />
        ) : (
          <div className="rv-chat">
            {data.history.map((m, i) => (
              <div key={i} className={`rv-msg rv-msg--${m.role}`}>
                <span className="rv-msg-role">{m.role === 'assistant' ? '面试官' : '我'}</span>
                <span className="rv-msg-text">{m.content}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function flatNodes(node: ThoughtNode): ThoughtNode[] {
  return [node, ...node.children.flatMap(flatNodes)];
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function InterviewReviewPage() {
  const [results, setResults]     = useState<ResultMeta[]>([]);
  const [selected, setSelected]   = useState<string | null>(null);
  const [loading, setLoading]     = useState(true);

  const fetchList = () => {
    setLoading(true);
    fetch(`${API_BASE}/interview/results`)
      .then(r => r.json())
      .then(d => { setResults(d.results ?? []); setLoading(false); })
      .catch(() => setLoading(false));
  };

  useEffect(() => {
    fetchList();
  }, []);

  useEffect(() => {
    if (results.length > 0 && !selected) setSelected(results[0].filename);
  }, [results]);

  const handleDelete = async (filename: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm(`删除 ${filename}？`)) return;
    await fetch(`${API_BASE}/interview/results/${filename}`, { method: 'DELETE' });
    setResults(prev => prev.filter(r => r.filename !== filename));
    if (selected === filename) setSelected(results.find(r => r.filename !== filename)?.filename ?? null);
  };

  return (
    <div className="rv-page">
      {/* Left sidebar */}
      <aside className="rv-sidebar">
        <div className="rv-sidebar-header">
          <span className="rv-sidebar-title">面试复盘</span>
          <button className="btn btn--sm btn--ghost" onClick={fetchList}>刷新</button>
        </div>

        {loading && <p className="rv-empty">加载中…</p>}
        {!loading && results.length === 0 && (
          <p className="rv-empty">还没有面试记录，完成一次模拟面试后会自动保存。</p>
        )}

        <div className="rv-list">
          {results.map(r => (
            <div
              key={r.filename}
              className={`rv-item ${selected === r.filename ? 'rv-item--active' : ''}`}
              onClick={() => setSelected(r.filename)}
            >
              <div className="rv-item-top">
                <span className="rv-item-date">{fmtDate(r.saved_at)}</span>
                <span className={`rv-item-state rv-item-state--${r.sm_state === 'DONE' ? 'done' : 'mid'}`}>
                  {r.sm_state}
                </span>
              </div>
              <div className="rv-item-middle">
                <span className="rv-item-dir">{r.direction || '未指定方向'}</span>
                <span className="rv-item-stats">
                  {r.task_count} 题
                  {r.avg_score !== null && ` · ${r.avg_score}分`}
                </span>
              </div>
              {r.jd_snippet && (
                <div className="rv-item-jd">{r.jd_snippet}</div>
              )}
              <button className="rv-item-delete" onClick={e => handleDelete(r.filename, e)} title="删除">✕</button>
            </div>
          ))}
        </div>
      </aside>

      {/* Right detail */}
      <main className="rv-main">
        {selected
          ? <SessionDetail filename={selected} />
          : <div className="rv-detail-empty">选择左侧记录查看详情</div>
        }
      </main>
    </div>
  );
}
