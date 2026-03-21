import { useState, useRef, useEffect, useCallback } from 'react';

const API_BASE = 'http://localhost:8000';

interface Source {
  source:   string;
  chapter:  string;
  chunk_id: string;
  text:     string;
}

interface QAMessage {
  id:      string;
  role:    'user' | 'assistant';
  content: string;
  sources?: Source[];
}

interface Note {
  note_id:    string;
  title:      string;
  size:       number;
  created_at: string;
  indexing?:  boolean;   // 乐观更新时显示"索引中"
}

// ── Source chip ────────────────────────────────────────────────────────────────

function SourceChip({ index, source, onOpen }: {
  index: number;
  source: Source;
  onOpen: (s: Source) => void;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <span className="qa-source-wrap">
      <button
        className="qa-source-chip"
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        onClick={() => onOpen(source)}
      >
        [{index + 1}] {source.chapter || source.source}
      </button>
      {hovered && (
        <div className="qa-source-tooltip">
          <div className="qa-source-tooltip-title">{source.source}</div>
          <div className="qa-source-tooltip-text">
            {source.text.slice(0, 220)}{source.text.length > 220 ? '…' : ''}
          </div>
        </div>
      )}
    </span>
  );
}

// ── Source viewer modal ────────────────────────────────────────────────────────

function SourceModal({ source, onClose }: { source: Source; onClose: () => void }) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [onClose]);
  return (
    <div className="source-modal-overlay" onClick={onClose}>
      <div className="source-modal" onClick={(e) => e.stopPropagation()}>
        <div className="source-modal-header">
          <div>
            <div className="source-modal-chapter">{source.chapter || '—'}</div>
            <div className="source-modal-source">{source.source}</div>
          </div>
          <button className="btn btn--icon btn--sm" onClick={onClose}>✕</button>
        </div>
        <pre className="source-modal-body">{source.text}</pre>
      </div>
    </div>
  );
}

// ── Notes sidebar (always visible) ────────────────────────────────────────────

function NotesSidebar({ notes, onDelete, onRefresh, onIndex }: {
  notes:     Note[];
  onDelete:  (note_id: string) => void;
  onRefresh: () => void;
  onIndex:   (note_id: string) => Promise<void>;
}) {
  const [expanded, setExpanded]         = useState<string | null>(null);
  const [expandedContent, setExpandedContent] = useState('');
  const [indexing, setIndexing]         = useState(false);

  const handleExpand = async (note_id: string) => {
    try {
      const res  = await fetch(`${API_BASE}/notes/${note_id}`);
      const data = await res.json();
      setExpandedContent(data.content ?? '');
      setExpanded(note_id);
    } catch { /* ignore */ }
  };

  const handleIndex = async () => {
    if (!expanded || indexing) return;
    setIndexing(true);
    try { await onIndex(expanded); } finally { setIndexing(false); }
  };

  const handleDelete = async (note_id: string) => {
    if (!confirm('确认删除这条笔记？')) return;
    await fetch(`${API_BASE}/notes/${note_id}`, { method: 'DELETE' });
    onDelete(note_id);
    if (expanded === note_id) setExpanded(null);
  };

  const fmtDate = (s: string) =>
    s.replace(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/, '$1-$2-$3 $4:$5');

  const expandedNote = notes.find((n) => n.note_id === expanded);

  // ── Expanded view ──
  if (expanded && expandedNote) {
    return (
      <aside className="notes-sidebar">
        <div className="note-expanded-header">
          <button className="btn btn--icon btn--sm" onClick={() => setExpanded(null)}>← 返回</button>
          <div className="note-expanded-title">{expandedNote.title}</div>
          <div className="note-expanded-meta">{fmtDate(expandedNote.created_at)}</div>
        </div>
        <pre className="note-expanded-content">{expandedContent}</pre>
        <div className="note-expanded-footer">
          <button
            className="btn btn--primary btn--sm"
            onClick={handleIndex}
            disabled={indexing}
            title="将笔记向量化加入知识库，之后问答时会参考此笔记"
          >
            {indexing ? '向量化中…' : '加入知识库'}
          </button>
          <button
            className="btn btn--icon btn--sm note-delete-btn"
            onClick={() => handleDelete(expanded)}
          >
            删除
          </button>
        </div>
      </aside>
    );
  }

  // ── List view ──
  return (
    <aside className="notes-sidebar">
      <div className="notes-sidebar-header">
        <span className="notes-sidebar-title">知识笔记</span>
        <button className="btn btn--icon btn--sm" onClick={onRefresh} title="刷新">↻</button>
      </div>
      <div className="notes-list">
        {notes.length === 0 && (
          <p className="notes-empty">暂无笔记，点击 ✦ 总结对话</p>
        )}
        {notes.map((n) => (
          <div key={n.note_id} className="note-item">
            <div className="note-item-header">
              <button
                className="note-item-title"
                onClick={() => !n.indexing && handleExpand(n.note_id)}
                disabled={n.indexing}
              >
                {n.title}
              </button>
              <button
                className="btn btn--icon btn--sm note-delete-btn"
                onClick={() => handleDelete(n.note_id)}
                disabled={n.indexing}
              >✕</button>
            </div>
            <div className="note-item-meta">
              {n.indexing
                ? <span className="note-indexing">保存中…</span>
                : fmtDate(n.created_at)}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────

export default function QAPage() {
  const [messages, setMessages]   = useState<QAMessage[]>([]);
  const [input, setInput]         = useState('');
  const [loading, setLoading]     = useState(false);
  const [viewer, setViewer]       = useState<Source | null>(null);
  const [summarizing, setSummarizing] = useState(false);
  const [notes, setNotes]         = useState<Note[]>([]);
  const historyRef = useRef<{ role: string; content: string }[]>([]);
  const bottomRef  = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchNotes = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/notes/list`);
      const data = await res.json();
      setNotes(data.notes ?? []);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => { fetchNotes(); }, [fetchNotes]);

  const submit = async () => {
    const question = input.trim();
    if (!question || loading) return;
    setInput('');

    const userMsg: QAMessage = { id: `u-${Date.now()}`, role: 'user', content: question };
    setMessages((prev) => [...prev, userMsg]);
    const aId = `a-${Date.now()}`;
    setMessages((prev) => [...prev, { id: aId, role: 'assistant', content: '', sources: [] }]);
    setLoading(true);

    const history = historyRef.current.slice(-12);
    try {
      const res = await fetch(`${API_BASE}/qa/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: question, history }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let fullText = '';
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (data === '[DONE]') continue;
          try {
            const payload = JSON.parse(data);
            if (payload.sources) {
              setMessages((prev) => prev.map((m) => m.id === aId ? { ...m, sources: payload.sources } : m));
            } else if (payload.text) {
              fullText += payload.text;
              setMessages((prev) => prev.map((m) => m.id === aId ? { ...m, content: fullText } : m));
            }
          } catch { /* ignore */ }
        }
      }
      historyRef.current = [...history,
        { role: 'user', content: question },
        { role: 'assistant', content: fullText },
      ];
    } catch {
      setMessages((prev) => prev.map((m) => m.id === aId
        ? { ...m, content: '请求失败，请检查后端是否运行' } : m));
    } finally {
      setLoading(false);
    }
  };

  const handleSummarize = () => {
    if (messages.length === 0 || summarizing) return;
    setSummarizing(true);

    // 立刻在侧边栏插入转圈占位
    const tempId = `temp_${Date.now()}`;
    setNotes((prev) => [{
      note_id:    tempId,
      title:      'AI 总结中…',
      size:       0,
      created_at: tempId.slice(5),
      indexing:   true,
    }, ...prev]);

    const msgs = messages
      .filter((m) => m.content)
      .map((m) => ({ role: m.role, content: m.content }));

    // 全程后台，不阻塞任何 UI
    (async () => {
      try {
        const sumRes = await fetch(`${API_BASE}/qa/summarize`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: msgs }),
        });
        const { title, content, questions = [] } = await sumRes.json();

        const saveRes = await fetch(`${API_BASE}/notes/save`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title, content, questions }),
        });
        const data = await saveRes.json();

        setNotes((prev) => prev.map((n) =>
          n.note_id === tempId
            ? { note_id: data.note_id, title, size: data.size, created_at: data.created_at }
            : n
        ));
      } catch {
        setNotes((prev) => prev.filter((n) => n.note_id !== tempId));
      } finally {
        setSummarizing(false);
      }
    })();
  };

  const handleIndexNote = async (note_id: string) => {
    await fetch(`${API_BASE}/notes/${note_id}/index`, { method: 'POST' });
  };

  return (
    <div className="qa-page">
      <NotesSidebar
        notes={notes}
        onDelete={(id) => setNotes((prev) => prev.filter((n) => n.note_id !== id))}
        onRefresh={fetchNotes}
        onIndex={handleIndexNote}
      />

      <div className="qa-main">
        <div className="qa-messages">
          {messages.length === 0 && (
            <div className="qa-empty">
              <p className="qa-empty-title">知识库问答</p>
              <p className="qa-empty-hint">问题会检索知识库，将相关内容送入 AI 上下文后回答</p>
            </div>
          )}
          {messages.map((msg) => (
            <div key={msg.id} className={`qa-msg qa-msg--${msg.role}`}>
              <div className="qa-msg-label">{msg.role === 'user' ? '你' : 'AI'}</div>
              <div className="qa-msg-bubble">
                {msg.content || (loading && msg.role === 'assistant'
                  ? <span className="stream-cursor" /> : null)}
              </div>
              {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                <div className="qa-sources">
                  <span className="qa-sources-label">参考资料</span>
                  {msg.sources.map((s, i) => (
                    <SourceChip key={s.chunk_id} index={i} source={s} onOpen={setViewer} />
                  ))}
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        <div className="qa-input-bar">
          <input
            className="qa-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); } }}
            placeholder="输入问题…"
            disabled={loading}
          />
          <button
            className="btn btn--icon"
            onClick={handleSummarize}
            disabled={summarizing || messages.length === 0}
            title="总结本次对话为知识点"
          >
            {summarizing ? '…' : '✦'}
          </button>
          <button className="btn btn--primary" onClick={submit} disabled={loading || !input.trim()}>
            {loading ? '…' : '发送'}
          </button>
        </div>
      </div>

      {viewer && <SourceModal source={viewer} onClose={() => setViewer(null)} />}
    </div>
  );
}
