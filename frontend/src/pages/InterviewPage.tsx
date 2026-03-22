import { useState, useCallback, useRef, useEffect } from 'react';
import { Message, InterviewState } from '../types';
import { useSpeechRecognition } from '../hooks/useSpeechRecognition';
import { useTTS } from '../hooks/useTTS';
import { InterviewerTile } from '../components/InterviewerTile';
import { UserTile } from '../components/UserTile';
import { StatusBadge } from '../components/StatusBadge';

const API_BASE = 'http://localhost:8000';
const SILENCE_TIMEOUT_MS = 800;

// ── Types ─────────────────────────────────────────────────────────────────────

interface ThoughtNode {
  id: string;
  node_type: 'task' | 'question';
  text: string;
  answer: string;
  depth: number;
  status: 'pending' | 'active' | 'asking' | 'answering' | 'answered' | 'scored' | 'done';
  score: number | null;
  verdict: 'pass' | 'continue' | 'deep_dive' | null;
  feedback: string;
  task_type: string;
  children: ThoughtNode[];
}

interface SMSnapshot {
  state: string;
  current_node_id: string | null;
  current_task_id: string | null;
  last_score: { score: number; verdict: string; feedback: string } | null;
}

type AgentState = 'idle' | 'running' | 'done';

interface AgentStatus {
  director:    { state: AgentState; note: string };
  scorer:      { state: AgentState; note: string };
  interviewer: { state: AgentState; note: string };
}

const INITIAL_AGENT_STATUS: AgentStatus = {
  director:    { state: 'idle', note: '等待开始' },
  scorer:      { state: 'idle', note: '待机' },
  interviewer: { state: 'idle', note: '等待开始' },
};

// SM states for the cycle diagram
const SM_ENTRY = ['INIT', 'PLANNING'] as const;
const SM_CYCLE = ['ASKING', 'ANSWERING', 'SCORING'] as const;

// ── SMPanel ───────────────────────────────────────────────────────────────────

function SMPanel({ sm, agentStatus }: { sm: SMSnapshot | null; agentStatus: AgentStatus }) {
  const agents = [
    { key: 'director',    icon: '🎬', label: '导演',   ...agentStatus.director },
    { key: 'interviewer', icon: '🎙', label: '面试官',  ...agentStatus.interviewer },
    { key: 'scorer',      icon: '📊', label: '评分员',  ...agentStatus.scorer },
  ] as const;

  const cur = sm?.state ?? 'INIT';
  const isDone = cur === 'DONE';
  const isDirecting = cur === 'DIRECTING';

  return (
    <div className="sm-panel">
      {/* Header */}
      <div className="sm-panel-header">
        <span className="im-panel-title">Agent 状态机</span>
        <span className={`sm-state-badge sm-state-badge--${isDone ? 'done' : SM_CYCLE.includes(cur as any) || isDirecting ? 'active' : 'idle'}`}>
          {cur}
        </span>
      </div>

      {/* Entry states strip */}
      <div className="sm-entry-strip">
        {SM_ENTRY.map((s, i) => {
          const entryOrder = ['INIT', 'PLANNING'];
          const cycleStarted = !entryOrder.includes(cur) && cur !== 'INIT';
          const isPast = cycleStarted || entryOrder.indexOf(cur) > entryOrder.indexOf(s);
          return (
            <span key={s} className={`sm-node sm-node--sm ${cur === s ? 'sm-node--cur' : isPast ? 'sm-node--past' : ''}`}>
              {s}
              {i < SM_ENTRY.length - 1 && <span className="sm-arrow-small">›</span>}
            </span>
          );
        })}
        <span className="sm-arrow-small">›</span>
        <span className={`sm-node sm-node--sm ${SM_CYCLE.includes(cur as any) || isDirecting ? 'sm-node--cur' : ''}`}>
          循环
        </span>
        {isDone && <><span className="sm-arrow-small">›</span><span className="sm-node sm-node--sm sm-node--done">DONE</span></>}
      </div>

      {/* Cycle loop */}
      <div className="sm-cycle">
        {SM_CYCLE.map((s, i) => (
          <span key={s} className="sm-cycle-item">
            <span className={`sm-node ${cur === s ? 'sm-node--cur' : ''}`}>{s}</span>
            {i < SM_CYCLE.length - 1 && <span className="sm-arrow">⟶</span>}
          </span>
        ))}
        <span className="sm-loop-arrow">↩</span>
      </div>
      {isDirecting && (
        <div className="sm-directing-note">
          <span className="sm-arrow-small">↳</span>
          <span className={`sm-node sm-node--cur`}>DIRECTING</span>
          <span className="sm-arrow-small">›</span>
          <span className="sm-node">ASKING</span>
        </div>
      )}

      {/* Agent rows */}
      <div className="sm-agents">
        {agents.map(a => (
          <div key={a.key} className={`sm-agent-row sm-agent-row--${a.state}`}>
            <span className="sm-agent-icon">{a.icon}</span>
            <span className="sm-agent-label">{a.label}</span>
            <span className="sm-agent-note">{a.note}</span>
            {a.state === 'running' && <span className="sm-spinner" />}
          </div>
        ))}
      </div>

      {/* Last score */}
      {sm?.last_score && (
        <div className="sm-last-score">
          <span className={`sm-score-num sm-score-num--${sm.last_score.score >= 4 ? 'good' : sm.last_score.score <= 2 ? 'low' : 'mid'}`}>
            {sm.last_score.score}/5
          </span>
          <span className={`sm-verdict sm-verdict--${sm.last_score.verdict}`}>{sm.last_score.verdict}</span>
          <span className="sm-score-fb">{sm.last_score.feedback}</span>
        </div>
      )}
    </div>
  );
}

// ── InfoPanel (思维树 + Chat tabs) ────────────────────────────────────────────

function InfoPanel({
  tree,
  sm,
  messages,
  interimTranscript,
  chatBodyRef,
  resultFile,
}: {
  tree: ThoughtNode[];
  sm: SMSnapshot | null;
  messages: Message[];
  interimTranscript: string;
  chatBodyRef: React.RefObject<HTMLDivElement>;
  resultFile: string | null;
}) {
  const [tab, setTab] = useState<'tree' | 'chat'>('tree');
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const toggleCollapse = (id: string) => {
    setCollapsed(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const renderNode = (node: ThoughtNode, indent = 0): React.ReactNode => {
    const isActiveNode = node.id === sm?.current_node_id || node.id === sm?.current_task_id;
    const isTask = node.node_type === 'task';
    const hasChildren = node.children.length > 0;
    const isCollapsed = collapsed.has(node.id);
    return (
      <div key={node.id}
        className={`info-task ${isActiveNode ? 'info-task--active' : ''} ${node.status === 'done' ? 'info-task--done' : ''} ${node.status === 'pending' ? 'info-task--pending' : ''} ${isTask ? 'info-task--task' : 'info-task--question'}`}
        style={{ marginLeft: indent * 14 }}
      >
        <div className="info-task-row" onClick={() => hasChildren && toggleCollapse(node.id)}
          style={{ cursor: hasChildren ? 'pointer' : 'default' }}>
          <span className="info-task-collapse">
            {hasChildren ? (isCollapsed ? '▸' : '▾') : ' '}
          </span>
          <span className="info-task-bullet">
            {isTask
              ? (node.status === 'done' ? '✓' : isActiveNode ? '▶' : '○')
              : (node.status === 'done' || node.status === 'scored' ? (node.score && node.score >= 4 ? '✓' : '·') : isActiveNode ? '❓' : '·')
            }
          </span>
          <span className="info-task-text">{node.text}</span>
          {node.score !== null && (
            <span className={`info-task-score info-task-score--${node.score >= 4 ? 'good' : node.score <= 2 ? 'low' : 'mid'}`}>
              {node.score}/5
            </span>
          )}
          {node.verdict && node.verdict !== 'pass' && (
            <span className={`info-task-verdict info-task-verdict--${node.verdict}`}>
              {node.verdict === 'deep_dive' ? '深挖' : '追问'}
            </span>
          )}
        </div>
        {node.feedback && (node.status === 'done' || node.status === 'scored') && (
          <div className="info-task-fb">{node.feedback}</div>
        )}
        {!isCollapsed && node.children.map(c => renderNode(c, indent + 1))}
      </div>
    );
  };

  return (
    <div className="info-panel">
      {/* Tab bar */}
      <div className="info-tabs">
        <button className={`info-tab ${tab === 'tree' ? 'info-tab--active' : ''}`} onClick={() => setTab('tree')}>
          思维树
        </button>
        <button className={`info-tab ${tab === 'chat' ? 'info-tab--active' : ''}`} onClick={() => setTab('chat')}>
          对话记录
        </button>
      </div>

      {/* Result download */}
      {resultFile && (
        <a className="info-result-link" href={`${API_BASE}/interview/results/${resultFile}`}
          target="_blank" rel="noreferrer">
          下载复盘 JSON ↓
        </a>
      )}

      {/* Tab content */}
      <div className="info-body">
        {tab === 'tree' ? (
          tree.length === 0
            ? <p className="info-empty">面试开始后显示思维树</p>
            : <>{tree.map(n => renderNode(n))}</>
        ) : (
          <div className="info-chat" ref={chatBodyRef}>
            {messages.length === 0
              ? <p className="info-empty">面试开始后显示对话…</p>
              : messages.map(m => (
                <div key={m.id} className={`info-msg info-msg--${m.role}`}>
                  <span className="info-msg-role">{m.role === 'assistant' ? '面试官' : '我'}</span>
                  <span className="info-msg-text">{m.content || <span className="thinking">…</span>}</span>
                </div>
              ))
            }
            {interimTranscript && (
              <div className="info-msg info-msg--user info-msg--interim">
                <span className="info-msg-role">我</span>
                <span className="info-msg-text">{interimTranscript}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── SetupOverlay ──────────────────────────────────────────────────────────────

function SetupOverlay({ loading, onStart }: { loading: boolean; onStart: (jd: string, dir: string) => void }) {
  const [jd, setJd]   = useState('');
  const [dir, setDir] = useState('');

  return (
    <div className="interview-overlay">
      <div className="im-setup-card">
        <h3 className="im-setup-title">开始模拟面试</h3>
        <p className="im-setup-hint">AI 导演会根据你的 Profile 和 JD 拆分具体考察任务</p>
        <div className="im-setup-field">
          <label>岗位 JD（可选）</label>
          <textarea rows={4} placeholder="粘贴岗位描述…" value={jd}
            onChange={e => setJd(e.target.value)} disabled={loading} />
        </div>
        <div className="im-setup-field">
          <label>考察方向（可选）</label>
          <input type="text" placeholder="例：操作系统 / 系统设计…" value={dir}
            onChange={e => setDir(e.target.value)} disabled={loading} />
        </div>
        <button
          className={`btn btn--primary btn--lg ${loading ? 'btn--disabled' : ''}`}
          disabled={loading} onClick={() => onStart(jd, dir)}
        >
          {loading ? '规划考察任务…' : '开始面试'}
        </button>
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function InterviewPage() {
  const [messages, setMessages]               = useState<Message[]>([]);
  const [state, setState]                     = useState<InterviewState>('idle');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [isStarted, setIsStarted]             = useState(false);
  const [loading, setLoading]                 = useState(false);
  const [error, setError]                     = useState<string | null>(null);
  const [sessionId, setSessionId]             = useState<string | null>(null);
  const [tree, setTree]                       = useState<ThoughtNode[]>([]);
  const [sm, setSm]                           = useState<SMSnapshot | null>(null);
  const [agentStatus, setAgentStatus]         = useState<AgentStatus>(INITIAL_AGENT_STATUS);
  const [resultFile, setResultFile]           = useState<string | null>(null);

  const sessionIdRef  = useRef<string | null>(null);
  const stateRef      = useRef<InterviewState>('idle');
  const accumulatedRef   = useRef('');
  const silenceTimerRef  = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ttsActionsRef    = useRef({ speak: (_: string) => {}, cancel: () => {} });
  const recognitionActionsRef = useRef({ start: () => {}, stop: () => {} });
  const chatBodyRef = useRef<HTMLDivElement>(null);

  useEffect(() => { stateRef.current = state; }, [state]);
  useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);
  useEffect(() => {
    chatBodyRef.current?.scrollTo({ top: chatBodyRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  // ── TTS ──────────────────────────────────────────────────────────────────────

  const { speak, cancel: cancelSpeak } = useTTS({
    lang: 'zh-CN', rate: 1,
    onStart: () => {
      setState('speaking');
      recognitionActionsRef.current.stop();
      setAgentStatus(s => ({ ...s, interviewer: { state: 'running', note: '说话中' } }));
    },
    onEnd: () => {
      if (stateRef.current !== 'idle') {
        setState('listening');
        recognitionActionsRef.current.start();
        setAgentStatus(s => ({ ...s, interviewer: { state: 'idle', note: '等待回答' } }));
      }
    },
  });
  useEffect(() => { ttsActionsRef.current = { speak, cancel: cancelSpeak }; }, [speak, cancelSpeak]);

  // ── Save session ──────────────────────────────────────────────────────────────

  const saveSession = useCallback(async () => {
    const sid = sessionIdRef.current;
    if (!sid) return;
    try {
      const res = await fetch(`${API_BASE}/interview/session/${sid}/save`, { method: 'POST' });
      const data = await res.json();
      if (data.filename) setResultFile(data.filename);
    } catch { /* ignore */ }
  }, []);

  // ── Submit turn ───────────────────────────────────────────────────────────────

  const submitTurn = useCallback(async (text: string) => {
    const sid = sessionIdRef.current;
    if (!sid || stateRef.current === 'processing') return;
    accumulatedRef.current = '';
    setInterimTranscript('');

    if (text) setMessages(prev => [...prev, { id: `u-${Date.now()}`, role: 'user', content: text, timestamp: new Date() }]);
    setState('processing');
    setAgentStatus(s => ({ ...s, interviewer: { state: 'running', note: '思考中' }, scorer: { state: 'running', note: '评分中' } }));

    const assistantId = `a-${Date.now()}`;
    setMessages(prev => [...prev, { id: assistantId, role: 'assistant' as const, content: '', timestamp: new Date() }]);

    try {
      const res = await fetch(`${API_BASE}/interview/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid, message: text }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '', fullText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (raw === '[DONE]') continue;
          try {
            const ev = JSON.parse(raw);
            if (ev.text !== undefined) {
              fullText += ev.text;
              setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: fullText } : m));
              setAgentStatus(s => ({ ...s, interviewer: { state: 'running', note: '回复中' } }));
            }
            if (ev.sm !== undefined) {
              const newSm: SMSnapshot = ev.sm;
              setSm(newSm);
              if (ev.tree) setTree(ev.tree);
              if (newSm.state === 'DONE') {
                setAgentStatus({
                  director:    { state: 'done', note: '面试结束' },
                  scorer:      { state: 'done', note: '评分完成' },
                  interviewer: { state: 'done', note: '面试结束' },
                });
                // 保存结果并获取文件名
                saveSession();
              } else if (newSm.state === 'ANSWERING' && newSm.last_score) {
                const ls = newSm.last_score;
                setAgentStatus(s => ({
                  ...s,
                  director:  { state: ls.verdict === 'pass' ? 'running' : 'idle', note: ls.verdict === 'pass' ? '推进下一任务' : '等待评分结果' },
                  scorer:    { state: 'done', note: `${ls.score}/5 · ${ls.verdict} · ${ls.feedback.slice(0, 18)}` },
                  interviewer: { state: 'idle', note: '等待回答' },
                }));
              }
            }
            if (ev.error) throw new Error(ev.error);
          } catch (e: any) { if (e?.message) throw e; }
        }
      }
      if (fullText) ttsActionsRef.current.speak(fullText);
      else setState('listening');
    } catch (err: any) {
      setError(err.message ?? '请求失败');
      setState('listening');
      setAgentStatus(s => ({ ...s, interviewer: { state: 'idle', note: '出错' }, scorer: { state: 'idle', note: '待机' } }));
    }
  }, [saveSession]);

  // ── Voice ─────────────────────────────────────────────────────────────────────

  const { start: startRecognition, stop: stopRecognition } = useSpeechRecognition({
    lang: 'zh-CN',
    onSpeechStart: useCallback(() => { if (stateRef.current !== 'processing') setState('listening'); }, []),
    onInterimResult: useCallback((t: string) => setInterimTranscript(accumulatedRef.current + t), []),
    onFinalResult: useCallback((t: string) => {
      accumulatedRef.current += t;
      setInterimTranscript(accumulatedRef.current);
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = setTimeout(() => {
        const txt = accumulatedRef.current.trim();
        if (txt) submitTurn(txt);
      }, SILENCE_TIMEOUT_MS);
    }, [submitTurn]),
    onError: (err) => setError(`语音识别错误: ${err}`),
  });
  useEffect(() => { recognitionActionsRef.current = { start: startRecognition, stop: stopRecognition }; }, [startRecognition, stopRecognition]);

  // ── Start / Stop ──────────────────────────────────────────────────────────────

  const handleStart = useCallback(async (jd: string, direction: string) => {
    setError(null);
    setLoading(true);
    setAgentStatus({ director: { state: 'running', note: '规划考察任务…' }, scorer: { state: 'idle', note: '待机' }, interviewer: { state: 'running', note: '生成开场问题…' } });
    try {
      const res = await fetch(`${API_BASE}/interview/start`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jd, direction }),
      });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || `HTTP ${res.status}`); }
      const data = await res.json();
      setSessionId(data.session_id);
      sessionIdRef.current = data.session_id;
      setTree(data.tree ?? []);
      setSm(data.sm);
      setAgentStatus(s => ({
        ...s,
        director:    { state: 'done', note: `已规划 ${(data.tree ?? []).filter((n: ThoughtNode) => n.node_type === 'task').length} 个任务` },
        interviewer: { state: 'idle', note: '等待回答' },
      }));

      // Show opening message directly (no extra API call needed)
      if (data.opening_message) {
        const msgId = `a-${Date.now()}`;
        setMessages([{ id: msgId, role: 'assistant', content: data.opening_message, timestamp: new Date() }]);
        ttsActionsRef.current.speak(data.opening_message);
      }

      setIsStarted(true);
      setState('listening');
      startRecognition();
    } catch (e: any) {
      setError(e.message ?? '启动失败');
      setAgentStatus(INITIAL_AGENT_STATUS);
    } finally {
      setLoading(false);
    }
  }, [startRecognition]);

  const handleStop = useCallback(async () => {
    setIsStarted(false); setState('idle'); stopRecognition();
    ttsActionsRef.current.cancel();
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    accumulatedRef.current = ''; setInterimTranscript('');
    await saveSession();
  }, [stopRecognition, saveSession]);

  const handleReset = useCallback(() => {
    setIsStarted(false); setState('idle'); setSessionId(null);
    setSm(null); setTree([]); setMessages([]);
    setAgentStatus(INITIAL_AGENT_STATUS); setError(null); setResultFile(null);
  }, []);

  const isDone = sm?.state === 'DONE';

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div className="interview-page">

      {/* Top bar */}
      <div className="interview-topbar">
        <StatusBadge state={state} />
        <div className="interview-topbar-actions">
          {isStarted && (
            <button className="btn btn--ghost btn--sm" onClick={saveSession} title="保存当前思维树">
              保存
            </button>
          )}
          {isStarted && !isDone && (
            <button className="btn btn--danger btn--sm" onClick={handleStop}>结束面试</button>
          )}
          {isDone && (
            <button className="btn btn--ghost btn--sm" onClick={handleReset}>重新开始</button>
          )}
        </div>
      </div>

      {/* 2×2 grid */}
      <div className="interview-grid">

        {/* ① 面试官 */}
        <div className="interview-tile interview-tile--interviewer">
          <InterviewerTile state={state} />
        </div>

        {/* ② 我 */}
        <div className="interview-tile interview-tile--user">
          <UserTile state={state} />
        </div>

        {/* ③ Agent 状态机 */}
        <div className="interview-tile interview-tile--sm">
          <SMPanel sm={sm} agentStatus={agentStatus} />
        </div>

        {/* ④ 后台信息 */}
        <div className="interview-tile interview-tile--info">
          <InfoPanel tree={tree} sm={sm} messages={messages}
            interimTranscript={interimTranscript} chatBodyRef={chatBodyRef}
            resultFile={resultFile} />
        </div>

        {/* Setup / Done overlay (inside grid, covers all 4 tiles) */}
        {!isStarted && <SetupOverlay loading={loading} onStart={handleStart} />}
        {isStarted && isDone && (
          <div className="interview-overlay">
            <div className="im-done-card">
              <div className="im-done-icon">🎉</div>
              <h3>面试结束</h3>
              <p>查看右下角评分详情</p>
              <button className="btn btn--ghost btn--sm" onClick={handleReset}>重新开始</button>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="error-bar">
          <span>{error}</span>
          <button className="btn btn--icon btn--sm" onClick={() => setError(null)}>✕</button>
        </div>
      )}
    </div>
  );
}
