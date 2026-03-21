import { useState, useCallback, useRef, useEffect } from 'react';
import { Message, InterviewState } from '../types';
import { useSpeechRecognition } from '../hooks/useSpeechRecognition';
import { useTTS } from '../hooks/useTTS';
import { InterviewerTile } from '../components/InterviewerTile';
import { UserTile } from '../components/UserTile';
import { StatusBadge } from '../components/StatusBadge';

const API_BASE = 'http://localhost:8000';
const SILENCE_TIMEOUT_MS = 800;

// ── 类型 ──────────────────────────────────────────────────────────────────────

interface TaskNode {
  id: string;
  topic: string;
  depth: number;
  status: 'pending' | 'active' | 'done';
  score: number | null;
  feedback: string;
  children: TaskNode[];
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

// ── TaskTree ──────────────────────────────────────────────────────────────────

function TaskTree({ tasks, currentId }: { tasks: TaskNode[]; currentId: string | null }) {
  const renderNode = (node: TaskNode, indent = 0) => {
    const isActive = node.id === currentId;
    return (
      <div key={node.id}
        className={`im-task-node ${isActive ? 'im-task-node--active' : ''} ${node.status === 'done' ? 'im-task-node--done' : ''} ${node.status === 'pending' ? 'im-task-node--pending' : ''}`}
        style={{ marginLeft: indent * 14 }}
      >
        <div className="im-task-node-row">
          <span className="im-task-bullet">
            {node.status === 'done' ? (node.score && node.score >= 4 ? '✓' : '·') : isActive ? '▶' : '○'}
          </span>
          <span className="im-task-topic">{node.topic}</span>
          {node.score !== null && (
            <span className={`im-task-score im-task-score--${node.score >= 4 ? 'good' : node.score <= 2 ? 'low' : 'mid'}`}>
              {node.score}/5
            </span>
          )}
        </div>
        {node.feedback && node.status === 'done' && (
          <div className="im-task-feedback">{node.feedback}</div>
        )}
        {node.children.map(c => renderNode(c, indent + 1))}
      </div>
    );
  };

  if (!tasks.length) return <p className="im-panel-empty">面试开始后显示考察路径</p>;
  return <>{tasks.map(n => renderNode(n))}</>;
}

// ── AgentStatusPanel ───────────────────────────────────────────────────────────

function AgentStatusPanel({ status }: { status: AgentStatus }) {
  const agents = [
    { key: 'director',    label: '导演',  icon: '🎬', ...status.director },
    { key: 'interviewer', label: '面试官', icon: '🎙', ...status.interviewer },
    { key: 'scorer',      label: '评分员', icon: '📊', ...status.scorer },
  ] as const;

  return (
    <>
      {agents.map(a => (
        <div key={a.key} className={`im-agent-row im-agent-row--${a.state}`}>
          <span className="im-agent-icon">{a.icon}</span>
          <span className="im-agent-label">{a.label}</span>
          <span className="im-agent-note">{a.note}</span>
          {a.state === 'running' && <span className="im-agent-spinner" />}
        </div>
      ))}
    </>
  );
}

// ── SetupOverlay ──────────────────────────────────────────────────────────────

function SetupOverlay({
  loading,
  onStart,
}: {
  loading: boolean;
  onStart: (jd: string, direction: string) => void;
}) {
  const [jd, setJd]             = useState('');
  const [direction, setDirection] = useState('');

  return (
    <div className="start-overlay">
      <div className="im-setup-card">
        <h3 className="im-setup-title">开始模拟面试</h3>
        <p className="im-setup-hint">
          填写岗位信息（可选），AI 导演会根据你的 Profile 规划考察路径
        </p>

        <div className="im-setup-field">
          <label>岗位 JD（可选）</label>
          <textarea
            placeholder="粘贴岗位描述…"
            rows={4}
            value={jd}
            onChange={e => setJd(e.target.value)}
            disabled={loading}
          />
        </div>

        <div className="im-setup-field">
          <label>考察方向（可选）</label>
          <input
            type="text"
            placeholder="例：操作系统 / 系统设计 / 数据库…"
            value={direction}
            onChange={e => setDirection(e.target.value)}
            disabled={loading}
          />
        </div>

        <button
          className={`btn btn--primary btn--lg ${loading ? 'btn--disabled' : ''}`}
          disabled={loading}
          onClick={() => onStart(jd, direction)}
        >
          {loading ? '规划面试方案…' : '开始面试'}
        </button>
      </div>
    </div>
  );
}

// ── 主页面 ────────────────────────────────────────────────────────────────────

export default function InterviewPage() {
  const [messages, setMessages]         = useState<Message[]>([]);
  const [state, setState]               = useState<InterviewState>('idle');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [isStarted, setIsStarted]       = useState(false);
  const [loading, setLoading]           = useState(false);
  const [error, setError]               = useState<string | null>(null);

  // Multi-agent state
  const [sessionId, setSessionId]         = useState<string | null>(null);
  const [tasks, setTasks]                 = useState<TaskNode[]>([]);
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [sessionStatus, setSessionStatus] = useState<'active' | 'done'>('active');
  const [agentStatus, setAgentStatus]     = useState<AgentStatus>(INITIAL_AGENT_STATUS);
  const [showChat, setShowChat]           = useState(false);

  const sessionIdRef    = useRef<string | null>(null);
  const stateRef        = useRef<InterviewState>('idle');
  const messagesRef     = useRef<Message[]>([]);
  const accumulatedRef  = useRef('');
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ttsActionsRef   = useRef({ speak: (_: string) => {}, cancel: () => {} });
  const recognitionActionsRef = useRef({ start: () => {}, stop: () => {} });
  const chatBodyRef     = useRef<HTMLDivElement>(null);

  useEffect(() => { stateRef.current = state; }, [state]);
  useEffect(() => { messagesRef.current = messages; }, [messages]);
  useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);

  // Auto-scroll chat
  useEffect(() => {
    chatBodyRef.current?.scrollTo({ top: chatBodyRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  // ── TTS ────────────────────────────────────────────────────────────────────

  const { speak, cancel: cancelSpeak } = useTTS({
    lang: 'zh-CN',
    rate: 1,
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

  useEffect(() => {
    ttsActionsRef.current = { speak, cancel: cancelSpeak };
  }, [speak, cancelSpeak]);

  // ── Submit turn to /interview/chat ─────────────────────────────────────────

  const submitTurn = useCallback(async (text: string) => {
    const sid = sessionIdRef.current;
    if (!sid || stateRef.current === 'processing') return;
    if (text) accumulatedRef.current = '';
    setInterimTranscript('');

    const userMsg: Message | null = text
      ? { id: `u-${Date.now()}`, role: 'user', content: text, timestamp: new Date() }
      : null;

    if (userMsg) {
      setMessages(prev => [...prev, userMsg]);
    }
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

      const reader  = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer    = '';
      let fullText  = '';

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
              setMessages(prev => prev.map(m =>
                m.id === assistantId ? { ...m, content: fullText } : m
              ));
              setAgentStatus(s => ({ ...s, interviewer: { state: 'running', note: '回复中' } }));
            }
            if (ev.tasks !== undefined) {
              setTasks(ev.tasks);
              setCurrentTaskId(ev.current_task_id ?? null);
              if (ev.session_status === 'done') {
                setSessionStatus('done');
                setAgentStatus({
                  director:    { state: 'done', note: '面试完成' },
                  scorer:      { state: 'done', note: '评分完成' },
                  interviewer: { state: 'done', note: '面试结束' },
                });
              } else {
                setAgentStatus(s => ({
                  ...s,
                  director: { state: 'idle', note: '已更新任务' },
                  scorer:   { state: 'done', note: '评分完成' },
                }));
              }
            }
            if (ev.error) throw new Error(ev.error);
          } catch (e: any) {
            if (e?.message) throw e;
          }
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

  // ── Voice recognition ─────────────────────────────────────────────────────

  const handleSpeechStart = useCallback(() => {
    if (stateRef.current !== 'processing') setState('listening');
  }, []);

  const handleInterimResult = useCallback((transcript: string) => {
    setInterimTranscript(accumulatedRef.current + transcript);
  }, []);

  const handleFinalResult = useCallback((transcript: string) => {
    accumulatedRef.current += transcript;
    setInterimTranscript(accumulatedRef.current);
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    silenceTimerRef.current = setTimeout(() => {
      const text = accumulatedRef.current.trim();
      if (text) submitTurn(text);
    }, SILENCE_TIMEOUT_MS);
  }, [submitTurn]);

  const { start: startRecognition, stop: stopRecognition } = useSpeechRecognition({
    lang: 'zh-CN',
    onSpeechStart: handleSpeechStart,
    onInterimResult: handleInterimResult,
    onFinalResult: handleFinalResult,
    onError: (err) => setError(`语音识别错误: ${err}`),
  });

  useEffect(() => {
    recognitionActionsRef.current = { start: startRecognition, stop: stopRecognition };
  }, [startRecognition, stopRecognition]);

  // ── Start interview ────────────────────────────────────────────────────────

  const handleStart = useCallback(async (jd: string, direction: string) => {
    setError(null);
    setLoading(true);
    setAgentStatus({
      director:    { state: 'running', note: '规划考察路径…' },
      scorer:      { state: 'idle',    note: '待机' },
      interviewer: { state: 'idle',    note: '等待导演…' },
    });

    try {
      const res = await fetch(`${API_BASE}/interview/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jd, direction }),
      });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || `HTTP ${res.status}`); }
      const data = await res.json();

      setSessionId(data.session_id);
      sessionIdRef.current = data.session_id;
      setTasks(data.tasks);
      setCurrentTaskId(data.current_task_id);
      setAgentStatus(s => ({
        ...s,
        director:    { state: 'done',    note: `已规划 ${data.tasks.length} 个话题` },
        interviewer: { state: 'running', note: '准备开场…' },
      }));

      setIsStarted(true);
      setState('processing');

      // 触发面试官开场
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

  // ── Stop interview ─────────────────────────────────────────────────────────

  const handleStop = useCallback(() => {
    setIsStarted(false);
    setState('idle');
    stopRecognition();
    ttsActionsRef.current.cancel();
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    accumulatedRef.current = '';
    setInterimTranscript('');
  }, [stopRecognition]);

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="interview-page">
      {/* 顶部状态栏 */}
      <div className="interview-topbar">
        <StatusBadge state={state} />
        <div className="interview-topbar-actions">
          <button
            className={`btn btn--icon ${showChat ? 'btn--active' : ''}`}
            onClick={() => setShowChat(v => !v)}
            title="对话记录"
          >
            💬
          </button>
        </div>
      </div>

      {/* 主内容 */}
      <div className="interview-content">

        {/* 视频区（左/主） */}
        <div className="video-area">
          <InterviewerTile state={state} />
          <UserTile state={state} />

          {!isStarted && (
            <SetupOverlay loading={loading} onStart={handleStart} />
          )}

          {isStarted && sessionStatus === 'done' && (
            <div className="start-overlay">
              <div className="im-done-card">
                <div className="im-done-icon">🎉</div>
                <h3>面试结束</h3>
                <p>查看右侧考察进度了解评分详情</p>
                <button className="btn btn--ghost btn--sm"
                  onClick={() => {
                    setIsStarted(false);
                    setState('idle');
                    setSessionId(null);
                    setTasks([]);
                    setCurrentTaskId(null);
                    setMessages([]);
                    setSessionStatus('active');
                    setAgentStatus(INITIAL_AGENT_STATUS);
                  }}>
                  重新开始
                </button>
              </div>
            </div>
          )}
        </div>

        {/* 右侧面板 */}
        <aside className="im-sidebar">

          {/* Agent 状态 */}
          <div className="im-panel">
            <div className="im-panel-header">
              <span className="im-panel-title">Agent 状态</span>
            </div>
            <div className="im-panel-body">
              <AgentStatusPanel status={agentStatus} />
            </div>
          </div>

          {/* 考察进度 */}
          <div className="im-panel im-panel--grow">
            <div className="im-panel-header">
              <span className="im-panel-title">考察进度</span>
              {sessionStatus === 'done' && <span className="badge badge--green">已完成</span>}
            </div>
            <div className="im-panel-body im-panel-body--scroll">
              <TaskTree tasks={tasks} currentId={currentTaskId} />
            </div>
          </div>

          {/* 对话记录（可折叠） */}
          {showChat && (
            <div className="im-panel im-panel--grow">
              <div className="im-panel-header">
                <span className="im-panel-title">对话记录</span>
                <button className="btn btn--icon btn--sm" onClick={() => setShowChat(false)}>✕</button>
              </div>
              <div className="im-panel-body im-panel-body--scroll im-chat-body" ref={chatBodyRef}>
                {messages.length === 0
                  ? <p className="im-panel-empty">面试开始后显示对话…</p>
                  : messages.map(m => (
                    <div key={m.id} className={`im-chat-msg im-chat-msg--${m.role}`}>
                      <span className="im-chat-role">
                        {m.role === 'assistant' ? '面试官' : '我'}
                      </span>
                      <span className="im-chat-text">
                        {m.content || <span className="thinking">…</span>}
                      </span>
                    </div>
                  ))
                }
                {interimTranscript && (
                  <div className="im-chat-msg im-chat-msg--user im-chat-interim">
                    <span className="im-chat-role">我</span>
                    <span className="im-chat-text">{interimTranscript}</span>
                  </div>
                )}
              </div>
            </div>
          )}

        </aside>
      </div>

      {/* 错误提示 */}
      {error && (
        <div className="error-bar">
          <span>{error}</span>
          <button className="btn btn--icon btn--sm" onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {/* 底部控制栏 */}
      {isStarted && sessionStatus === 'active' && (
        <footer className="footer">
          <button className="btn btn--danger btn--lg" onClick={handleStop}>结束面试</button>
        </footer>
      )}
    </div>
  );
}
