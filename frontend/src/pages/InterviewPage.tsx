import { useState, useRef, useCallback, useEffect } from 'react';

const API_BASE = 'http://localhost:8000';

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

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

type PageState = 'setup' | 'starting' | 'chat' | 'done';

// ── TaskTree 组件 ─────────────────────────────────────────────────────────────

function TaskTree({ tasks, currentId }: { tasks: TaskNode[]; currentId: string | null }) {
  if (!tasks.length) return null;

  const renderNode = (node: TaskNode, indent = 0) => {
    const isActive  = node.id === currentId;
    const statusCls = isActive ? 'task-active' : node.status === 'done' ? 'task-done' : 'task-pending';
    const scoreStr  = node.score !== null ? ` ${node.score}/5` : '';

    return (
      <div key={node.id} className={`task-node task-node--depth${Math.min(indent, 2)} ${statusCls}`}>
        <div className="task-node-header">
          <span className="task-node-bullet">
            {node.status === 'done' ? (node.score && node.score >= 4 ? '✓' : node.score && node.score <= 2 ? '▾' : '·') : isActive ? '▶' : '○'}
          </span>
          <span className="task-node-topic">{node.topic}</span>
          {node.score !== null && (
            <span className={`task-score task-score--${node.score >= 4 ? 'good' : node.score <= 2 ? 'low' : 'mid'}`}>
              {scoreStr}
            </span>
          )}
        </div>
        {node.feedback && node.status === 'done' && (
          <div className="task-node-feedback">{node.feedback}</div>
        )}
        {node.children.length > 0 && (
          <div className="task-children">
            {node.children.map(c => renderNode(c, indent + 1))}
          </div>
        )}
      </div>
    );
  };

  return <div className="task-tree">{tasks.map(n => renderNode(n))}</div>;
}

// ── SetupPanel 组件 ───────────────────────────────────────────────────────────

function SetupPanel({
  onStart,
  loading,
}: {
  onStart: (jd: string, direction: string) => void;
  loading: boolean;
}) {
  const [jd, setJd] = useState('');
  const [direction, setDirection] = useState('');

  return (
    <div className="interview-setup">
      <h2 className="interview-setup-title">开始模拟面试</h2>
      <p className="interview-setup-hint">
        填写 JD 和考察方向（可选），面试官会根据你的 Profile 和岗位要求规划问题。
      </p>

      <div className="interview-setup-field">
        <label>岗位 JD（可选）</label>
        <textarea
          placeholder="粘贴岗位描述，面试官会更有针对性…"
          value={jd}
          onChange={e => setJd(e.target.value)}
          rows={5}
          disabled={loading}
        />
      </div>

      <div className="interview-setup-field">
        <label>指定考察方向（可选）</label>
        <input
          type="text"
          placeholder="例如：操作系统 / 数据库 / 系统设计…"
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
        {loading ? '规划面试方案中…' : '开始面试'}
      </button>
    </div>
  );
}

// ── 主页面 ────────────────────────────────────────────────────────────────────

export default function InterviewPage() {
  const [pageState, setPageState] = useState<PageState>('setup');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [tasks, setTasks] = useState<TaskNode[]>([]);
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ── 开始面试 ────────────────────────────────────────────────────────────────

  const handleStart = useCallback(async (jd: string, direction: string) => {
    setError(null);
    setPageState('starting');

    try {
      const res = await fetch(`${API_BASE}/interview/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jd, direction }),
      });
      if (!res.ok) {
        const e = await res.json();
        throw new Error(e.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setSessionId(data.session_id);
      setTasks(data.tasks);
      setCurrentTaskId(data.current_task_id);
      setPageState('chat');

      // 立即触发面试官开场
      await sendMessage(data.session_id, '');
    } catch (e: any) {
      setError(e.message ?? '启动失败');
      setPageState('setup');
    }
  }, []);

  // ── 发送消息 ────────────────────────────────────────────────────────────────

  const sendMessage = useCallback(async (sid: string, userText: string) => {
    setSending(true);

    if (userText.trim()) {
      const userMsg: ChatMessage = {
        id: `u-${Date.now()}`,
        role: 'user',
        content: userText.trim(),
      };
      setMessages(prev => [...prev, userMsg]);
    }

    const assistantId = `a-${Date.now()}`;
    setMessages(prev => [...prev, { id: assistantId, role: 'assistant', content: '' }]);

    try {
      const res = await fetch(`${API_BASE}/interview/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid, message: userText }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

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
              setMessages(prev =>
                prev.map(m =>
                  m.id === assistantId ? { ...m, content: m.content + ev.text } : m
                )
              );
            }

            if (ev.tasks !== undefined) {
              setTasks(ev.tasks);
              setCurrentTaskId(ev.current_task_id ?? null);
              if (ev.session_status === 'done') {
                setPageState('done');
              }
            }

            if (ev.error) {
              setError(ev.error);
            }
          } catch { /* ignore malformed */ }
        }
      }
    } catch (e: any) {
      setError(e.message ?? '请求失败');
    } finally {
      setSending(false);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, []);

  const handleSend = useCallback(() => {
    const text = input.trim();
    if (!text || sending || !sessionId) return;
    setInput('');
    sendMessage(sessionId, text);
  }, [input, sending, sessionId, sendMessage]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ── 渲染 ────────────────────────────────────────────────────────────────────

  if (pageState === 'setup' || pageState === 'starting') {
    return (
      <div className="interview-v2-page interview-v2-page--setup">
        <SetupPanel onStart={handleStart} loading={pageState === 'starting'} />
        {error && (
          <div className="error-bar">
            <span>{error}</span>
            <button className="btn btn--icon btn--sm" onClick={() => setError(null)}>✕</button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="interview-v2-page">
      {/* 左侧：任务树 */}
      <aside className="interview-sidebar">
        <div className="interview-sidebar-header">
          <span className="interview-sidebar-title">考察进度</span>
          {pageState === 'done' && <span className="badge badge--green">已完成</span>}
        </div>
        <TaskTree tasks={tasks} currentId={currentTaskId} />
        {pageState === 'done' && (
          <button
            className="btn btn--ghost btn--sm interview-restart-btn"
            onClick={() => {
              setPageState('setup');
              setMessages([]);
              setTasks([]);
              setCurrentTaskId(null);
              setSessionId(null);
            }}
          >
            重新开始
          </button>
        )}
      </aside>

      {/* 右侧：聊天区 */}
      <main className="interview-chat">
        <div className="interview-messages">
          {messages.map(m => (
            <div key={m.id} className={`interview-bubble interview-bubble--${m.role}`}>
              <div className="interview-bubble-label">
                {m.role === 'assistant' ? '面试官' : '我'}
              </div>
              <div className="interview-bubble-text">{m.content || <span className="thinking">…</span>}</div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {error && (
          <div className="error-bar">
            <span>{error}</span>
            <button className="btn btn--icon btn--sm" onClick={() => setError(null)}>✕</button>
          </div>
        )}

        {pageState !== 'done' && (
          <div className="interview-input-row">
            <textarea
              ref={inputRef}
              className="interview-input"
              placeholder="输入你的回答… (Enter 发送，Shift+Enter 换行)"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={sending}
              rows={3}
            />
            <button
              className={`btn btn--primary interview-send-btn ${sending || !input.trim() ? 'btn--disabled' : ''}`}
              disabled={sending || !input.trim()}
              onClick={handleSend}
            >
              {sending ? '…' : '发送'}
            </button>
          </div>
        )}

        {pageState === 'done' && (
          <div className="interview-done-banner">
            面试结束！请查看左侧评分详情。
          </div>
        )}
      </main>
    </div>
  );
}
