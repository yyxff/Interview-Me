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

interface TaskNode {
  id: string;
  task: string;
  task_type: string;
  depth: number;
  status: 'pending' | 'active' | 'done';
  score: number | null;
  feedback: string;
  children: TaskNode[];
}

interface SMSnapshot {
  state: string;
  current_task_id: string | null;
  last_score: { score: number; feedback: string } | null;
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

// SM cycling states (for the loop diagram)
const SM_CYCLE = ['INTERVIEWING', 'SCORING', 'DIRECTING'] as const;
const SM_ENTRY = ['INIT', 'PLANNING', 'READY'] as const;

// ── SMPanel ───────────────────────────────────────────────────────────────────

function SMPanel({ sm, agentStatus }: { sm: SMSnapshot | null; agentStatus: AgentStatus }) {
  const agents = [
    { key: 'director',    icon: '🎬', label: '导演',   ...agentStatus.director },
    { key: 'interviewer', icon: '🎙', label: '面试官',  ...agentStatus.interviewer },
    { key: 'scorer',      icon: '📊', label: '评分员',  ...agentStatus.scorer },
  ] as const;

  const cur = sm?.state ?? 'INIT';
  const isDone = cur === 'DONE';

  return (
    <div className="sm-panel">
      {/* Header */}
      <div className="sm-panel-header">
        <span className="im-panel-title">Agent 状态机</span>
        <span className={`sm-state-badge sm-state-badge--${isDone ? 'done' : SM_CYCLE.includes(cur as any) ? 'active' : 'idle'}`}>
          {cur}
        </span>
      </div>

      {/* Entry states strip */}
      <div className="sm-entry-strip">
        {SM_ENTRY.map((s, i) => (
          <span key={s} className={`sm-node sm-node--sm ${cur === s ? 'sm-node--cur' : s === 'READY' || s === 'PLANNING' || s === 'INIT' ? (
            ['PLANNING','READY','INTERVIEWING','SCORING','DIRECTING','DONE'].indexOf(cur) >
            ['INIT','PLANNING','READY'].indexOf(s) ? 'sm-node--past' : '') : ''}`}>
            {s}
            {i < SM_ENTRY.length - 1 && <span className="sm-arrow-small">›</span>}
          </span>
        ))}
        <span className="sm-arrow-small">›</span>
        <span className={`sm-node sm-node--sm ${SM_CYCLE.includes(cur as any) ? 'sm-node--cur' : ''}`}>
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
          <span className="sm-score-fb">{sm.last_score.feedback}</span>
        </div>
      )}
    </div>
  );
}

// ── InfoPanel (Tasks + Chat tabs) ─────────────────────────────────────────────

function InfoPanel({
  tasks,
  sm,
  messages,
  interimTranscript,
  chatBodyRef,
}: {
  tasks: TaskNode[];
  sm: SMSnapshot | null;
  messages: Message[];
  interimTranscript: string;
  chatBodyRef: React.RefObject<HTMLDivElement>;
}) {
  const [tab, setTab] = useState<'tasks' | 'chat'>('tasks');

  const renderTask = (node: TaskNode, indent = 0): React.ReactNode => {
    const isActive = node.id === sm?.current_task_id;
    return (
      <div key={node.id}
        className={`info-task ${isActive ? 'info-task--active' : ''} ${node.status === 'done' ? 'info-task--done' : ''} ${node.status === 'pending' ? 'info-task--pending' : ''}`}
        style={{ marginLeft: indent * 14 }}
      >
        <div className="info-task-row">
          <span className="info-task-bullet">
            {node.status === 'done' ? (node.score && node.score >= 4 ? '✓' : '·') : isActive ? '▶' : '○'}
          </span>
          <span className="info-task-text">{node.task}</span>
          {node.score !== null && (
            <span className={`info-task-score info-task-score--${node.score >= 4 ? 'good' : node.score <= 2 ? 'low' : 'mid'}`}>
              {node.score}/5
            </span>
          )}
        </div>
        {node.feedback && node.status === 'done' && (
          <div className="info-task-fb">{node.feedback}</div>
        )}
        {node.children.map(c => renderTask(c, indent + 1))}
      </div>
    );
  };

  return (
    <div className="info-panel">
      {/* Tab bar */}
      <div className="info-tabs">
        <button className={`info-tab ${tab === 'tasks' ? 'info-tab--active' : ''}`} onClick={() => setTab('tasks')}>
          考察进度
        </button>
        <button className={`info-tab ${tab === 'chat' ? 'info-tab--active' : ''}`} onClick={() => setTab('chat')}>
          对话记录
        </button>
      </div>

      {/* Tab content */}
      <div className="info-body">
        {tab === 'tasks' ? (
          tasks.length === 0
            ? <p className="info-empty">面试开始后显示考察路径</p>
            : <>{tasks.map(n => renderTask(n))}</>
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
  const [jd, setJd]         = useState('');
  const [dir, setDir]       = useState('');

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
  const [tasks, setTasks]                     = useState<TaskNode[]>([]);
  const [sm, setSm]                           = useState<SMSnapshot | null>(null);
  const [agentStatus, setAgentStatus]         = useState<AgentStatus>(INITIAL_AGENT_STATUS);

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

  // ── Submit turn ───────────────────────────────────────────────────────────────

  const submitTurn = useCallback(async (text: string) => {
    const sid = sessionIdRef.current;
    if (!sid || stateRef.current === 'processing') return;
    accumulatedRef.current = '';
    setInterimTranscript('');

    if (text) setMessages(prev => [...prev, { id: `u-${Date.now()}`, role: 'user', content: text, timestamp: new Date() }]);
    setState('processing');
    setAgentStatus(s => ({ ...s, interviewer: { state: 'running', note: '思考中' } }));

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
              if (ev.tasks) setTasks(ev.tasks);
              if (newSm.state === 'DONE') {
                setAgentStatus({
                  director:    { state: 'done', note: '面试结束' },
                  scorer:      { state: 'done', note: '评分完成' },
                  interviewer: { state: 'done', note: '面试结束' },
                });
              } else if (newSm.state === 'INTERVIEWING' && newSm.last_score) {
                setAgentStatus(s => ({
                  ...s,
                  director:  { state: 'idle', note: '已更新任务路径' },
                  scorer:    { state: 'done', note: `${newSm.last_score!.score}/5 · ${newSm.last_score!.feedback.slice(0,18)}` },
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
      setAgentStatus(s => ({ ...s, interviewer: { state: 'idle', note: '出错' } }));
    }
  }, []);

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
    setAgentStatus({ director: { state: 'running', note: '规划考察任务…' }, scorer: { state: 'idle', note: '待机' }, interviewer: { state: 'idle', note: '等待导演…' } });
    try {
      const res = await fetch(`${API_BASE}/interview/start`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jd, direction }),
      });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || `HTTP ${res.status}`); }
      const data = await res.json();
      setSessionId(data.session_id);
      sessionIdRef.current = data.session_id;
      setTasks(data.tasks);
      setSm(data.sm);
      setAgentStatus(s => ({ ...s, director: { state: 'done', note: `已规划 ${data.tasks.length} 个任务` }, interviewer: { state: 'running', note: '开场中…' } }));
      setIsStarted(true);
      setState('processing');
      await submitTurn('');
      setState('listening');
      startRecognition();
    } catch (e: any) {
      setError(e.message ?? '启动失败');
      setAgentStatus(INITIAL_AGENT_STATUS);
    } finally {
      setLoading(false);
    }
  }, [submitTurn, startRecognition]);

  const handleStop = useCallback(() => {
    setIsStarted(false); setState('idle'); stopRecognition();
    ttsActionsRef.current.cancel();
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    accumulatedRef.current = ''; setInterimTranscript('');
  }, [stopRecognition]);

  const handleReset = useCallback(() => {
    setIsStarted(false); setState('idle'); setSessionId(null);
    setSm(null); setTasks([]); setMessages([]);
    setAgentStatus(INITIAL_AGENT_STATUS); setError(null);
  }, []);

  const isDone = sm?.state === 'DONE';

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div className="interview-page">

      {/* Top bar */}
      <div className="interview-topbar">
        <StatusBadge state={state} />
        <div className="interview-topbar-actions">
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
          <InfoPanel tasks={tasks} sm={sm} messages={messages}
            interimTranscript={interimTranscript} chatBodyRef={chatBodyRef} />
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
