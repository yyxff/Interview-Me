import { useState, useRef, useEffect, useCallback } from 'react';

const API_BASE = 'http://localhost:8000';

// ── Shared types ───────────────────────────────────────────────────────────────

interface KnowledgeFile {
  name:     string;
  filename: string;
  size:     number;
  indexed:  boolean;
}

interface IndexProgress {
  status:        'idle' | 'running' | 'done' | 'error';
  file:          string;
  chunks_done:   number;
  chunks_total:  number;
  vectors_added: number;
  elapsed_s:     number;
  eta_s:         number | null;
  error:         string | null;
}

interface Source {
  source:   string;
  chapter:  string;
  chunk_id: string;
  text:     string;
}

interface QAMessage {
  id:       string;
  role:     'user' | 'assistant';
  content:  string;
  sources?: Source[];
}

interface Note {
  note_id:    string;
  title:      string;
  size:       number;
  created_at: string;
  indexing?:  boolean;
  indexed?:   boolean;   // 已加入知识库
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function fmtSeconds(s: number | null): string {
  if (s === null || s < 0) return '…';
  if (s < 60) return `${Math.round(s)}s`;
  return `${Math.floor(s / 60)}m${Math.round(s % 60)}s`;
}

function fmtDate(s: string): string {
  return s.replace(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/, '$1-$2-$3 $4:$5');
}

// ── Knowledge sidebar (left) ───────────────────────────────────────────────────

function KnowledgeSidebar({ indexedNotes }: { indexedNotes: Note[] }) {
  const [files, setFiles]         = useState<KnowledgeFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [progress, setProgress]   = useState<IndexProgress | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollRef      = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchList = async () => {
    try {
      const res  = await fetch(`${API_BASE}/knowledge/list`);
      const data = await res.json();
      setFiles(data.files ?? []);
    } catch { /* ignore */ }
  };

  const fetchProgress = async () => {
    try {
      const res  = await fetch(`${API_BASE}/rag/index-progress`);
      const data: IndexProgress = await res.json();
      setProgress(data);
      if (data.status === 'done' || data.status === 'error') fetchList();
    } catch { /* ignore */ }
  };

  useEffect(() => {
    fetchProgress();
    pollRef.current = setInterval(fetchProgress, 2000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  useEffect(() => { fetchList(); }, []);

  const handleUpload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.epub')) {
      setUploadError('目前只支持 EPUB 格式');
      return;
    }
    setUploading(true);
    setUploadError(null);
    const form = new FormData();
    form.append('file', file);
    try {
      const res = await fetch(`${API_BASE}/upload/knowledge`, { method: 'POST', body: form });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || `HTTP ${res.status}`); }
      await fetchList();
    } catch (e: any) {
      setUploadError(e.message ?? '上传失败');
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const isIndexing = progress?.status === 'running';
  const pct = progress && progress.chunks_total > 0
    ? Math.round(progress.chunks_done / progress.chunks_total * 100) : 0;

  return (
    <aside className="kqa-left">
      <div className="kqa-panel-header">
        <span className="kqa-panel-title">知识库</span>
        <label className={`btn btn--sm btn--primary ${uploading || isIndexing ? 'btn--disabled' : ''}`}>
          {uploading ? '处理中…' : '+ 上传'}
          <input ref={fileInputRef} type="file" accept=".epub" style={{ display: 'none' }}
            disabled={uploading || isIndexing}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) handleUpload(f); }} />
        </label>
      </div>

      {isIndexing && progress && (
        <div className="index-progress">
          <div className="index-progress-bar">
            <div className="index-progress-fill" style={{ width: `${pct}%` }} />
          </div>
          <div className="index-progress-text">
            <span>向量化 {progress.chunks_done}/{progress.chunks_total}</span>
            <span>剩余 {fmtSeconds(progress.eta_s)}</span>
          </div>
          <div className="index-progress-file">{progress.file}</div>
        </div>
      )}
      {progress?.status === 'done' && progress.vectors_added > 0 && (
        <div className="index-done">索引完成，共 {progress.vectors_added} 个向量</div>
      )}
      {progress?.status === 'error' && (
        <div className="knowledge-upload-error">索引出错: {progress.error}</div>
      )}
      {uploadError && <p className="knowledge-upload-error">{uploadError}</p>}

      <div className="kqa-file-list">
        {files.length === 0 && !uploading && (
          <p className="kqa-empty">还没有文件，上传 EPUB 开始构建知识库</p>
        )}
        {files.map((f) => (
          <div key={f.name} className="kqa-file-item">
            <span className="kqa-file-name">{f.name}</span>
            <span className="kqa-file-meta">
              {(f.size / 1024).toFixed(0)}K
              <span className={`badge ${f.indexed ? 'badge--green' : isIndexing ? 'badge--amber' : 'badge--gray'}`}>
                {f.indexed ? '已索引' : isIndexing ? '索引中' : '未索引'}
              </span>
            </span>
          </div>
        ))}

        {indexedNotes.length > 0 && (
          <>
            <div className="kqa-section-divider">知识笔记</div>
            {indexedNotes.map((n) => (
              <div key={n.note_id} className="kqa-file-item kqa-file-item--note">
                <span className="kqa-file-name">{n.title}</span>
                <span className="kqa-file-meta">
                  <span className="badge badge--green">已索引</span>
                </span>
              </div>
            ))}
          </>
        )}
      </div>
    </aside>
  );
}

// ── Source chip ────────────────────────────────────────────────────────────────

function SourceChip({ index, source, onOpen }: {
  index:  number;
  source: Source;
  onOpen: (s: Source) => void;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <span className="qa-source-wrap">
      <button className="qa-source-chip"
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

// ── Notes sidebar (right) ─────────────────────────────────────────────────────

function NotesSidebar({ notes, onDelete, onRefresh, onIndex }: {
  notes:     Note[];
  onDelete:  (note_id: string) => void;
  onRefresh: () => void;
  onIndex:   (note_id: string) => Promise<void>;
}) {
  const [expanded, setExpanded]           = useState<string | null>(null);
  const [expandedContent, setExpandedContent] = useState('');
  const [indexing, setIndexing]           = useState(false);
  const [indexStatus, setIndexStatus]     = useState<'idle' | 'ok' | 'error'>('idle');

  const handleExpand = async (note_id: string) => {
    try {
      const res  = await fetch(`${API_BASE}/notes/${note_id}`);
      const data = await res.json();
      setExpandedContent(data.content ?? '');
      setExpanded(note_id);
      setIndexStatus('idle');
    } catch { /* ignore */ }
  };

  const handleIndex = async () => {
    if (!expanded || indexing) return;
    setIndexing(true);
    setIndexStatus('idle');
    try {
      await onIndex(expanded);
      setIndexStatus('ok');
      setTimeout(() => setIndexStatus('idle'), 2500);
    } catch {
      setIndexStatus('error');
      setTimeout(() => setIndexStatus('idle'), 2500);
    } finally {
      setIndexing(false);
    }
  };

  const handleDelete = async (note_id: string) => {
    if (!confirm('确认删除这条笔记？')) return;
    await fetch(`${API_BASE}/notes/${note_id}`, { method: 'DELETE' });
    onDelete(note_id);
    if (expanded === note_id) setExpanded(null);
  };

  const expandedNote = notes.find((n) => n.note_id === expanded);

  // ── Expanded view ──
  if (expanded && expandedNote) {
    return (
      <aside className="kqa-right">
        <div className="note-expanded-header">
          <button className="btn btn--icon btn--sm" onClick={() => setExpanded(null)}>← 返回</button>
          <div className="note-expanded-title">
            {expandedNote.title}
            {expandedNote.indexed && <span className="note-indexed-badge" title="已加入知识库">知识库</span>}
          </div>
          <div className="note-expanded-meta">{fmtDate(expandedNote.created_at)}</div>
        </div>
        <pre className="note-expanded-content">{expandedContent}</pre>
        <div className="note-expanded-footer">
          <button
            className={`btn btn--sm ${indexStatus === 'ok' ? 'btn--success' : indexStatus === 'error' ? 'btn--danger' : 'btn--primary'}`}
            onClick={handleIndex}
            disabled={indexing || indexStatus === 'ok'}
            title="将笔记向量化加入知识库，之后问答时会参考此笔记"
          >
            {indexing ? '向量化中…' : indexStatus === 'ok' ? '✓ 已加入' : indexStatus === 'error' ? '失败，重试' : '加入知识库'}
          </button>
          <button className="btn btn--icon btn--sm note-delete-btn" onClick={() => handleDelete(expanded)}>
            删除
          </button>
        </div>
      </aside>
    );
  }

  // ── List view ──
  return (
    <aside className="kqa-right">
      <div className="kqa-panel-header">
        <span className="kqa-panel-title">知识笔记</span>
        <button className="btn btn--icon btn--sm" onClick={onRefresh} title="刷新">↻</button>
      </div>
      <div className="notes-list">
        {notes.length === 0 && (
          <p className="kqa-empty">暂无笔记，点击 ✦ 总结对话</p>
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
              <div className="note-item-actions">
                {n.indexed && <span className="note-indexed-badge" title="已加入知识库">知识库</span>}
                <button
                  className="btn btn--icon btn--sm note-delete-btn"
                  onClick={() => handleDelete(n.note_id)}
                  disabled={n.indexing}
                >✕</button>
              </div>
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

export default function KnowledgeQAPage() {
  const [messages, setMessages]       = useState<QAMessage[]>([]);
  const [input, setInput]             = useState('');
  const [loading, setLoading]         = useState(false);
  const [viewer, setViewer]           = useState<Source | null>(null);
  const [summarizing, setSummarizing] = useState(false);
  const [notes, setNotes]             = useState<Note[]>([]);
  const historyRef = useRef<{ role: string; content: string }[]>([]);
  const bottomRef  = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchNotes = useCallback(async () => {
    try {
      const res  = await fetch(`${API_BASE}/notes/list`);
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

      const reader  = res.body!.getReader();
      const decoder = new TextDecoder();
      let fullText = '';
      let buffer   = '';
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
        { role: 'user',      content: question  },
        { role: 'assistant', content: fullText  },
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
    const res = await fetch(`${API_BASE}/notes/${note_id}/index`, { method: 'POST' });
    if (!res.ok) throw new Error('index failed');
    // 乐观更新：标记为已索引（后台线程完成后真正生效）
    setNotes((prev) => prev.map((n) => n.note_id === note_id ? { ...n, indexed: true } : n));
  };

  return (
    <div className="kqa-page">
      <KnowledgeSidebar indexedNotes={notes.filter((n) => n.indexed && !n.indexing)} />

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

      <NotesSidebar
        notes={notes}
        onDelete={(id) => setNotes((prev) => prev.filter((n) => n.note_id !== id))}
        onRefresh={fetchNotes}
        onIndex={handleIndexNote}
      />

      {viewer && <SourceModal source={viewer} onClose={() => setViewer(null)} />}
    </div>
  );
}
