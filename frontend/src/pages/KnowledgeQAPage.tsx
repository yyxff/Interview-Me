import React, { useState, useRef, useEffect, useCallback, useMemo, Fragment } from 'react';
import ReactMarkdown from 'react-markdown';
import ForceGraph3D from '3d-force-graph';

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

interface GraphIndexProgress {
  status:       'idle' | 'running' | 'done' | 'error' | 'unavailable';
  source:       string;
  chunks_done:  number;
  chunks_total: number;
  entities:     number;
  relations:    number;
  elapsed_s:    number;
  eta_s:        number | null;
  concurrency:  number;
  error:        string | null;
}

interface Source {
  source:     string;
  path?:      string;
  chapter:    string;
  chunk_id:   string;
  text:       string;
  via_graph?: boolean;
}

interface GraphNode {
  id:               string;
  entity_type:      string;
  description:      string;
  source_chunk_ids: string[];
  used:             boolean;
  adjacent:         boolean;
}

interface GraphEdge {
  source:    string;
  target:    string;
  predicate: string;
  used:      boolean;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

interface QAMessage {
  id:              string;
  role:            'user' | 'assistant';
  content:         string;
  sources?:        Source[];
  graph?:          GraphData;
  rewritten_query?: string;
}

// ── Forked conversation tree ───────────────────────────────────────────────────

interface ConvNode {
  id:       string;
  parentId: string | null;
  question: QAMessage;
  answer:   QAMessage;
  childIds: string[];
}

interface ConvTab {
  id:        string;
  label:     string;
  currentId: string | null;
}

// Git-branch icon for the fork button
function BranchIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="4"  cy="12" r="1.6" fill="currentColor"/>
      <circle cx="4"  cy="5"  r="1.6" fill="currentColor"/>
      <circle cx="10" cy="2"  r="1.6" fill="currentColor"/>
      <line x1="4" y1="6.6" x2="4" y2="10.4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      <path d="M4,5 C4,3 4,2 10,2" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
    </svg>
  );
}



function getPath(nodes: Record<string, ConvNode>, leafId: string | null): ConvNode[] {
  const path: ConvNode[] = [];
  let cur: string | null = leafId;
  while (cur) {
    const n = nodes[cur];
    if (!n) break;
    path.unshift(n);
    cur = n.parentId;
  }
  return path;
}

// ── Branch map: mind-map style SVG visualization ──────────────────────────────

const BM_NODE_W = 82;
const BM_NODE_H = 26;
const BM_V_STEP = 64;  // vertical step per depth level (px)
const BM_H_SLOT = 100; // horizontal space per "slot" (px)
const BM_PAD    = 10;  // canvas padding

interface BMapNode { id: string; x: number; y: number; label: string; isActive: boolean; isPath: boolean; }
interface BMapEdge { d: string; isPath: boolean; }

function layoutBranchMap(
  nodes: Record<string, ConvNode>,
  rootIds: string[],
  activePath: Set<string>,
  activeId: string | null,
): { mnodes: BMapNode[]; edges: BMapEdge[]; svgW: number; svgH: number } {
  const mnodes: BMapNode[] = [];
  const edges:  BMapEdge[] = [];

  // Count leaf slots in a subtree
  function slots(id: string): number {
    const n = nodes[id];
    if (!n || n.childIds.length === 0) return 1;
    return n.childIds.reduce((s, cid) => s + slots(cid), 0);
  }

  function place(id: string, depth: number, slotStart: number) {
    const n = nodes[id];
    if (!n) return;
    const totalSlots = slots(id);
    const y = BM_PAD + depth * BM_V_STEP;
    const x = BM_PAD + (slotStart + (totalSlots - 1) / 2) * BM_H_SLOT;
    const raw = n.question.content;
    mnodes.push({
      id, x, y,
      label: raw.length > 12 ? raw.slice(0, 12) + '…' : raw,
      isActive: id === activeId,
      isPath: activePath.has(id),
    });
    let cs = slotStart;
    for (const cid of n.childIds) {
      const cSlots = slots(cid);
      const cy = BM_PAD + (depth + 1) * BM_V_STEP;
      const cx = BM_PAD + (cs + (cSlots - 1) / 2) * BM_H_SLOT;
      const x1 = x  + BM_NODE_W / 2, y1 = y  + BM_NODE_H;  // bottom-center of parent
      const x2 = cx + BM_NODE_W / 2, y2 = cy;               // top-center of child
      const my = (y1 + y2) / 2;
      const isPathEdge = activePath.has(id) && activePath.has(cid);
      edges.push({ d: `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`, isPath: isPathEdge });
      place(cid, depth + 1, cs);
      cs += cSlots;
    }
  }

  let rootSlot = 0;
  for (const rid of rootIds) {
    place(rid, 0, rootSlot);
    rootSlot += slots(rid) + 0.6; // small gap between separate root trees
  }

  const svgW = mnodes.reduce((m, n) => Math.max(m, n.x + BM_NODE_W + BM_PAD), 160);
  const svgH = mnodes.reduce((m, n) => Math.max(m, n.y + BM_NODE_H + BM_PAD), 60);
  return { mnodes, edges, svgW, svgH };
}

function BranchMap({ nodes, rootIds, activeCurrentId, onNavigate }: {
  nodes:           Record<string, ConvNode>;
  rootIds:         string[];
  activeCurrentId: string | null;
  onNavigate:      (id: string) => void;
}) {
  const [tooltip, setTooltip]       = useState<{ x: number; y: number; text: string } | null>(null);
  const [pan, setPan]               = useState<{ x: number; y: number }>({ x: 10, y: 10 });
  const [isDragging, setIsDragging] = useState(false);
  const dragRef = useRef<{ startX: number; startY: number; panX: number; panY: number } | null>(null);
  const didDragRef = useRef(false);

  const activePath = useMemo(() => {
    const s = new Set<string>();
    let cur: string | null = activeCurrentId;
    while (cur) { s.add(cur); cur = nodes[cur]?.parentId ?? null; }
    return s;
  }, [nodes, activeCurrentId]);

  const { mnodes, edges, svgW, svgH } = useMemo(
    () => layoutBranchMap(nodes, rootIds, activePath, activeCurrentId),
    [nodes, rootIds, activePath, activeCurrentId],
  );

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    dragRef.current = { startX: e.clientX, startY: e.clientY, panX: pan.x, panY: pan.y };
    didDragRef.current = false;
    setIsDragging(true);
    e.preventDefault();
  }, [pan]);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.startX;
    const dy = e.clientY - dragRef.current.startY;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) didDragRef.current = true;
    setPan({ x: dragRef.current.panX + dx, y: dragRef.current.panY + dy });
  }, []);

  const onMouseUp = useCallback(() => {
    dragRef.current = null;
    setIsDragging(false);
  }, []);

  const handleNodeClick = useCallback((id: string) => {
    if (!didDragRef.current) onNavigate(id);
  }, [onNavigate]);

  if (rootIds.length === 0) return <p className="bpanel-empty">开始对话后显示分支图</p>;

  return (
    <div
      className={`branch-map-wrap${isDragging ? ' branch-map-wrap--dragging' : ''}`}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
    >
      <div className="branch-map-pan" style={{ transform: `translate(${pan.x}px, ${pan.y}px)` }}>
        <svg width={svgW} height={svgH}>
          {edges.map((e, i) => (
            <path key={i} d={e.d} className={`bmap-edge${e.isPath ? ' bmap-edge--path' : ''}`} />
          ))}
          {mnodes.map(n => (
            <g
              key={n.id}
              className="bmap-node-g"
              onClick={() => handleNodeClick(n.id)}
              onMouseEnter={ev => setTooltip({ x: ev.clientX, y: ev.clientY, text: nodes[n.id]?.question.content ?? '' })}
              onMouseLeave={() => setTooltip(null)}
            >
              <rect x={n.x} y={n.y} width={BM_NODE_W} height={BM_NODE_H} rx={5}
                className={`bmap-node${n.isActive ? ' bmap-node--active' : n.isPath ? ' bmap-node--path' : ''}`} />
              <text x={n.x + BM_NODE_W / 2} y={n.y + BM_NODE_H / 2 + 4} textAnchor="middle"
                className={`bmap-text${n.isActive ? ' bmap-text--active' : n.isPath ? ' bmap-text--path' : ''}`}>
                {n.label}
              </text>
            </g>
          ))}
        </svg>
      </div>
      {tooltip && !isDragging && (
        <div className="bmap-tooltip" style={{ position: 'fixed', left: tooltip.x + 14, top: tooltip.y - 10 }}>
          {tooltip.text}
        </div>
      )}
    </div>
  );
}

// ── QA Session persistence ─────────────────────────────────────────────────────

interface QASessionMeta {
  session_id: string;
  title:      string;
  created_at: string;
  updated_at: string;
  node_count: number;
}

function SessionListView({ onLoad, onNew }: {
  onLoad: (sid: string) => void;
  onNew:  () => void;
}) {
  const [sessions, setSessions] = useState<QASessionMeta[]>([]);
  const [loading, setLoading]   = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/qa-sessions`)
      .then(r => r.json())
      .then(d => setSessions(d.sessions ?? []))
      .finally(() => setLoading(false));
  }, []);

  const handleDelete = async (sid: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('确认删除此对话记录？')) return;
    await fetch(`${API_BASE}/qa-sessions/${sid}`, { method: 'DELETE' });
    setSessions(prev => prev.filter(s => s.session_id !== sid));
  };

  return (
    <div className="qa-session-list-page">
      <div className="qa-session-list-header">
        <span className="qa-session-list-title">知识库问答</span>
        <button className="btn btn--primary" onClick={onNew}>新对话</button>
      </div>
      {loading ? (
        <div className="qa-session-empty">加载中…</div>
      ) : sessions.length === 0 ? (
        <div className="qa-session-empty">暂无对话记录，点击「新对话」开始</div>
      ) : (
        <div className="qa-session-list">
          {sessions.map(s => (
            <div key={s.session_id} className="qa-session-item" onClick={() => onLoad(s.session_id)}>
              <div className="qa-session-item-title">{s.title || '新对话'}</div>
              <div className="qa-session-item-meta">
                <span>{fmtDate(s.updated_at)}</span>
                <span>{s.node_count} 条对话</span>
              </div>
              <button className="qa-session-item-del" onClick={e => handleDelete(s.session_id, e)}>×</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
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

interface ProfileStatus { uploaded: boolean; size: number; sections: string[]; }

function KnowledgeSidebar({ indexedNotes }: { indexedNotes: Note[] }) {
  const [files, setFiles]               = useState<KnowledgeFile[]>([]);
  const [uploading, setUploading]       = useState(false);
  const [uploadError, setUploadError]   = useState<string | null>(null);
  const [progress, setProgress]         = useState<IndexProgress | null>(null);
  const [graphProgress, setGraphProgress] = useState<GraphIndexProgress | null>(null);
  const [profile, setProfile]           = useState<ProfileStatus | null>(null);
  const [profileUploading, setProfileUploading] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);
  const fileInputRef    = useRef<HTMLInputElement>(null);
  const profileInputRef = useRef<HTMLInputElement>(null);
  const pollRef         = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchList = async () => {
    try {
      const res  = await fetch(`${API_BASE}/knowledge/list`);
      const data = await res.json();
      setFiles(data.files ?? []);
    } catch { /* ignore */ }
  };

  const fetchProgress = async () => {
    try {
      const [r1, r2] = await Promise.all([
        fetch(`${API_BASE}/rag/index-progress`),
        fetch(`${API_BASE}/graph/index-progress`),
      ]);
      const d1: IndexProgress      = await r1.json();
      const d2: GraphIndexProgress = await r2.json();
      setProgress(d1);
      setGraphProgress(d2);
      if (d1.status === 'done' || d1.status === 'error') fetchList();
    } catch { /* ignore */ }
  };

  useEffect(() => {
    fetchProgress();
    pollRef.current = setInterval(fetchProgress, 2000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  useEffect(() => { fetchList(); }, []);

  const fetchProfile = async () => {
    try {
      const res = await fetch(`${API_BASE}/profile/status`);
      setProfile(await res.json());
    } catch { /* ignore */ }
  };
  useEffect(() => { fetchProfile(); }, []);

  const handleProfileUpload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.md')) {
      setProfileError('只支持 .md 文件');
      return;
    }
    setProfileUploading(true);
    setProfileError(null);
    const form = new FormData();
    form.append('file', file);
    try {
      const res = await fetch(`${API_BASE}/profile/upload`, { method: 'POST', body: form });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || `HTTP ${res.status}`); }
      await fetchProfile();
    } catch (e: any) {
      setProfileError(e.message ?? '上传失败');
    } finally {
      setProfileUploading(false);
      if (profileInputRef.current) profileInputRef.current.value = '';
    }
  };

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

  const isIndexing      = progress?.status === 'running';
  const isGraphIndexing = graphProgress?.status === 'running';
  const pct = progress && progress.chunks_total > 0
    ? Math.round(progress.chunks_done / progress.chunks_total * 100) : 0;
  const graphPct = graphProgress && graphProgress.chunks_total > 0
    ? Math.round(graphProgress.chunks_done / graphProgress.chunks_total * 100) : 0;

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
          <div className="index-progress-label">向量索引</div>
          <div className="index-progress-bar">
            <div className="index-progress-fill" style={{ width: `${pct}%` }} />
          </div>
          <div className="index-progress-text">
            <span>{progress.chunks_done}/{progress.chunks_total} chunks</span>
            <span>剩余 {fmtSeconds(progress.eta_s)}</span>
          </div>
          <div className="index-progress-file">{progress.file}</div>
        </div>
      )}
      {progress?.status === 'done' && progress.vectors_added > 0 && (
        <div className="index-done">向量索引完成，共 {progress.vectors_added} 个向量</div>
      )}
      {progress?.status === 'error' && (
        <div className="knowledge-upload-error">向量索引出错: {progress.error}</div>
      )}

      {isGraphIndexing && graphProgress && (
        <div className="index-progress index-progress--graph">
          <div className="index-progress-label">图谱索引</div>
          <div className="index-progress-bar">
            <div className="index-progress-fill index-progress-fill--graph" style={{ width: `${graphPct}%` }} />
          </div>
          <div className="index-progress-text">
            <span>{graphProgress.chunks_done}/{graphProgress.chunks_total} chunks · 并发 {graphProgress.concurrency}</span>
            <span>剩余 {fmtSeconds(graphProgress.eta_s)}</span>
          </div>
          <div className="index-progress-file">
            {graphProgress.source}
            {graphProgress.entities > 0 && ` · ${graphProgress.entities} 实体 ${graphProgress.relations} 关系`}
          </div>
        </div>
      )}
      {graphProgress?.status === 'done' && graphProgress.entities > 0 && (
        <div className="index-done index-done--graph">
          图谱索引完成，{graphProgress.entities} 实体 · {graphProgress.relations} 关系
        </div>
      )}
      {graphProgress?.status === 'error' && (
        <div className="knowledge-upload-error">图谱索引出错: {graphProgress.error}</div>
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

      {/* Profile 区域 */}
      <div className="profile-section">
        <div className="profile-section-header">
          <span className="profile-section-title">个人 Profile</span>
          <span className={`badge ${profile?.uploaded ? 'badge--green' : 'badge--gray'}`}>
            {profile?.uploaded ? `${(profile.size / 1024).toFixed(1)}K` : '未上传'}
          </span>
          <label className={`btn btn--sm ${profileUploading ? 'btn--disabled' : 'btn--ghost'}`}>
            {profileUploading ? '上传中…' : '上传 MD'}
            <input ref={profileInputRef} type="file" accept=".md" style={{ display: 'none' }}
              disabled={profileUploading}
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleProfileUpload(f); }} />
          </label>
        </div>
        {profileError && <p className="knowledge-upload-error">{profileError}</p>}
        {profile?.uploaded && profile.sections.length > 0 && (
          <ul className="profile-sections">
            {profile.sections.map(s => <li key={s}>{s}</li>)}
          </ul>
        )}
      </div>
    </aside>
  );
}

// ── Graph visualization ────────────────────────────────────────────────────────

interface SimNode extends GraphNode {
  x: number; y: number;
  vx: number; vy: number;
}

const REPULSION    = 3000;
const SPRING_LEN   = 120;
const SPRING_K     = 0.04;
const DAMPING      = 0.82;
const GRAVITY      = 0.008;
const NODE_R       = 18;
const LABEL_MAX    = 8;  // max chars in node label before truncating

const COLOR_USED     = '#6366f1';
const COLOR_ADJ      = '#374151';
const COLOR_USED_TXT = '#e2e8f0';
const COLOR_ADJ_TXT  = '#9ca3af';
const COLOR_EDGE_USED= '#6366f1';
const COLOR_EDGE_ADJ = '#374151';
const COLOR_BG       = '#0d0d16';

function GraphPanel({ data, sources, onOpen }: {
  data:    GraphData;
  sources: Source[];
  onOpen:  (s: Source) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simRef    = useRef<SimNode[]>([]);
  const animRef   = useRef<number>(0);
  const hoverRef    = useRef<SimNode | null>(null);
  const [tooltip, setTooltip]   = useState<{x:number;y:number;node:SimNode}|null>(null);
  const [hovering, setHovering] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.nodes.length === 0) return;
    const W = canvas.width;
    const H = canvas.height;

    // init positions
    simRef.current = data.nodes.map((n, i) => ({
      ...n,
      x: W / 2 + Math.cos(i / data.nodes.length * Math.PI * 2) * 140,
      y: H / 2 + Math.sin(i / data.nodes.length * Math.PI * 2) * 140,
      vx: 0, vy: 0,
    }));

    const byId = new Map(simRef.current.map(n => [n.id, n]));

    function drawArrow(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number) {
      const dx = x2 - x1, dy = y2 - y1;
      const len = Math.sqrt(dx * dx + dy * dy) || 1;
      const ux = dx / len, uy = dy / len;
      // shorten to node edge
      const sx = x1 + ux * NODE_R, sy = y1 + uy * NODE_R;
      const ex = x2 - ux * NODE_R, ey = y2 - uy * NODE_R;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(ex, ey);
      ctx.stroke();
      // arrow head
      const ax = ex - ux * 8 + uy * 5;
      const ay = ey - uy * 8 - ux * 5;
      const bx = ex - ux * 8 - uy * 5;
      const by = ey - uy * 8 + ux * 5;
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.closePath();
      ctx.fill();
    }

    function tick() {
      const nodes = simRef.current;

      // repulsion
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[j].x - nodes[i].x;
          const dy = nodes[j].y - nodes[i].y;
          const d2 = dx * dx + dy * dy || 1;
          const d  = Math.sqrt(d2);
          const f  = REPULSION / d2;
          nodes[i].vx -= f * dx / d; nodes[i].vy -= f * dy / d;
          nodes[j].vx += f * dx / d; nodes[j].vy += f * dy / d;
        }
      }
      // spring
      for (const e of data.edges) {
        const a = byId.get(e.source), b = byId.get(e.target);
        if (!a || !b) continue;
        const dx = b.x - a.x, dy = b.y - a.y;
        const d  = Math.sqrt(dx * dx + dy * dy) || 1;
        const f  = SPRING_K * (d - SPRING_LEN);
        a.vx += f * dx / d; a.vy += f * dy / d;
        b.vx -= f * dx / d; b.vy -= f * dy / d;
      }
      // gravity + damping + clamp
      for (const n of nodes) {
        n.vx += (W / 2 - n.x) * GRAVITY;
        n.vy += (H / 2 - n.y) * GRAVITY;
        n.vx *= DAMPING; n.vy *= DAMPING;
        n.x = Math.max(NODE_R + 2, Math.min(W - NODE_R - 2, n.x + n.vx));
        n.y = Math.max(NODE_R + 2, Math.min(H - NODE_R - 2, n.y + n.vy));
      }

      // draw
      if (!canvas) return;
      const ctx = canvas.getContext('2d')!;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = COLOR_BG;
      ctx.fillRect(0, 0, W, H);

      // edges
      for (const e of data.edges) {
        const a = byId.get(e.source), b = byId.get(e.target);
        if (!a || !b) continue;
        const color = e.used ? COLOR_EDGE_USED : COLOR_EDGE_ADJ;
        ctx.strokeStyle = color;
        ctx.fillStyle   = color;
        ctx.lineWidth   = e.used ? 1.5 : 0.8;
        ctx.globalAlpha = e.used ? 1 : 0.35;
        drawArrow(ctx, a.x, a.y, b.x, b.y);
        // label
        if (e.used) {
          const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
          ctx.globalAlpha = 0.85;
          ctx.fillStyle = '#a5b4fc';
          ctx.font = '9px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(e.predicate, mx, my - 4);
        }
        ctx.globalAlpha = 1;
      }

      // nodes
      for (const n of nodes) {
        const isHov = hoverRef.current?.id === n.id;
        const fill  = n.used ? COLOR_USED : COLOR_ADJ;
        const alpha = n.adjacent ? 0.45 : 1;
        ctx.globalAlpha = isHov ? 1 : alpha;
        ctx.beginPath();
        ctx.arc(n.x, n.y, isHov ? NODE_R + 3 : NODE_R, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.fill();
        if (n.used) {
          ctx.strokeStyle = '#818cf8';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
        const label = n.id.length > LABEL_MAX ? n.id.slice(0, LABEL_MAX) + '…' : n.id;
        ctx.fillStyle = n.used ? COLOR_USED_TXT : COLOR_ADJ_TXT;
        ctx.font = `${n.used ? 'bold ' : ''}10px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.globalAlpha = isHov ? 1 : alpha;
        ctx.fillText(label, n.x, n.y);
        ctx.globalAlpha = 1;
      }

      animRef.current = requestAnimationFrame(tick);
    }

    animRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animRef.current);
  }, [data]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width  / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top)  * scaleY;
    const found = simRef.current.find(
      n => Math.hypot(n.x - mx, n.y - my) < NODE_R + 4
    ) ?? null;
    hoverRef.current = found;
    setHovering(!!found);
    if (found) {
      setTooltip({ x: e.clientX, y: e.clientY, node: found });
    } else {
      setTooltip(null);
    }
  };

  const handleClick = async (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width  / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top)  * scaleY;
    const node = simRef.current.find(n => Math.hypot(n.x - mx, n.y - my) < NODE_R + 4);
    if (!node) return;
    // 优先从已有 sources 中找
    const chunkIds = node.source_chunk_ids ?? [];
    if (chunkIds.length === 0) return;
    const matched = sources.find(s => chunkIds.includes(s.chunk_id));
    if (matched) { onOpen(matched); return; }
    try {
      const res = await fetch(`/chunk/${encodeURIComponent(chunkIds[0])}`);
      if (!res.ok) return;
      const data = await res.json();
      onOpen({ ...data, via_graph: true });
    } catch { /* ignore */ }
  };

  if (data.nodes.length === 0) return null;

  return (
    <div className="graph-panel">
      <div className="graph-panel-header">
        <span className="graph-panel-title">知识图谱</span>
        <span className="graph-panel-meta">
          {data.nodes.filter(n => n.used).length} 命中 · {data.nodes.filter(n => n.adjacent).length} 邻居 · {data.edges.length} 关系
        </span>
      </div>
      <div className="graph-canvas-wrap">
        <canvas
          ref={canvasRef}
          width={700}
          height={340}
          className="graph-canvas"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => { hoverRef.current = null; setHovering(false); setTooltip(null); }}
          onClick={handleClick}
          style={{ cursor: hovering ? 'pointer' : 'default' }}
        />
        {tooltip && (
          <div className="graph-tooltip" style={{ left: tooltip.x + 12, top: tooltip.y - 8, position: 'fixed' }}>
            <div className="graph-tooltip-name">{tooltip.node.id}</div>
            <div className="graph-tooltip-type">{tooltip.node.entity_type}</div>
            {tooltip.node.description && (
              <div className="graph-tooltip-desc">{tooltip.node.description.slice(0, 120)}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Full knowledge graph viewer (3D) ──────────────────────────────────────────

interface FullNode {
  id:          string;
  entity_type: string;
  description: string;
  source:      string;
  color?:      string;
}
interface FullLink {
  source:    string;
  target:    string;
  predicate: string;
}

function sourceColor(source: string, allSources: string[]): string {
  const idx = allSources.indexOf(source);
  const hue = Math.round((idx / (allSources.length || 1)) * 330); // avoid red wraparound
  return `hsl(${hue},65%,60%)`;
}

function nodeLabel(n: FullNode): string {
  return `<div style="background:#1a1a2e;padding:8px 12px;border-radius:6px;font-family:sans-serif;max-width:220px;border:1px solid #374151">
    <div style="font-weight:600;color:#e5e7eb;font-size:13px">${n.id}</div>
    <div style="color:#6366f1;font-size:10px;margin-top:2px">${n.entity_type}</div>
    <div style="color:#6b7280;font-size:10px">${n.source}</div>
    ${n.description ? `<div style="color:#9ca3af;font-size:11px;margin-top:4px;line-height:1.4">${n.description.slice(0, 120)}</div>` : ''}
  </div>`;
}

function FullGraphViewer({ onClose }: { onClose: () => void }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef     = useRef<any>(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState('');
  const [legend,  setLegend]  = useState<{source: string; color: string}[]>([]);
  const [stats,   setStats]   = useState('');

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Mount point outside React's reconciler so ForceGraph3D's DOM
    // mutations never conflict with React's removeChild calls.
    const mountEl = document.createElement('div');
    mountEl.style.cssText = 'width:100%;height:100%;position:absolute;inset:0';
    container.appendChild(mountEl);

    let destroyed = false;

    // Defer via setTimeout so React Strict Mode's synchronous
    // cleanup (mount→cleanup→mount) clears the timer before
    // any WebGL context is created, avoiding context-loss on second mount.
    const timerId = window.setTimeout(() => {
      if (destroyed) return;

      fetch(`${API_BASE}/graph/full`)
        .then(r => r.json())
        .then((data: { nodes: FullNode[]; edges: FullLink[]; error?: string }) => {
          if (destroyed) return;
          if (data.error) { setError(data.error); setLoading(false); return; }

          const allSources = [...new Set(data.nodes.map(n => n.source))].sort();
          const colorMap: Record<string, string> = {};
          allSources.forEach(s => { colorMap[s] = sourceColor(s, allSources); });

          setLegend(allSources.map(s => ({ source: s, color: colorMap[s] })));
          setStats(`${data.nodes.length} 节点 · ${data.edges.length} 关系 · ${allSources.length} 来源`);

          const nodes = data.nodes.map(n => ({ ...n, color: colorMap[n.source] ?? '#6366f1' }));
          const links = data.edges;

          const Graph = new ForceGraph3D(mountEl)
            .backgroundColor('#0d0d16')
            .width(mountEl.clientWidth)
            .height(mountEl.clientHeight)
            .nodeColor((n: any) => n.color)
            .nodeLabel((n: any) => nodeLabel(n))
            .nodeRelSize(4)
            .linkColor(() => 'rgba(148,150,255,0.8)')
            .linkWidth(1.5)
            .linkLabel((l: any) => l.predicate)
            .linkDirectionalArrowLength(3)
            .linkDirectionalArrowRelPos(1)
            .linkDirectionalParticles(1)
            .linkDirectionalParticleSpeed(0.004)
            .onNodeDragEnd((node: any) => {
              node.fx = node.x;
              node.fy = node.y;
              node.fz = node.z;
            })
            .graphData({ nodes, links });

          graphRef.current = Graph;
          setLoading(false);

          const ro = new ResizeObserver(() => {
            Graph.width(mountEl.clientWidth).height(mountEl.clientHeight);
          });
          ro.observe(mountEl);
          (Graph as any)._ro = ro;
        })
        .catch(e => { if (!destroyed) { setError(String(e)); setLoading(false); } });
    }, 0);

    return () => {
      destroyed = true;
      clearTimeout(timerId);
      (graphRef.current as any)?._ro?.disconnect();
      graphRef.current?._destructor?.();
      graphRef.current = null;
      if (container.contains(mountEl)) container.removeChild(mountEl);
    };
  }, []);

  return (
    <div className="fullgraph-overlay">
      <div className="fullgraph-header">
        <span className="fullgraph-title">知识图谱全览</span>
        {stats && <span className="fullgraph-stats">{stats}</span>}
        <span className="fullgraph-hint">左键旋转 · 右键平移 · 滚轮缩放 · 节点可拖拽固定</span>
        <button className="fullgraph-close" onClick={onClose}>✕</button>
      </div>

      {legend.length > 0 && (
        <div className="fullgraph-legend">
          {legend.map(l => (
            <span key={l.source} className="fullgraph-legend-item">
              <span className="fullgraph-legend-dot" style={{ background: l.color }} />
              {l.source}
            </span>
          ))}
        </div>
      )}

      <div ref={containerRef} className="fullgraph-canvas-wrap">
        {loading && <div className="fullgraph-loading">加载图谱中…</div>}
        {error   && <div className="fullgraph-error">错误：{error}</div>}
      </div>
    </div>
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
      <button className={`qa-source-chip${source.via_graph ? ' qa-source-chip--graph' : ''}`}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        onClick={() => onOpen(source)}
      >
        [{index + 1}] {source.path || source.chapter || source.source}{source.via_graph ? ' ✦' : ''}
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

// ── Inline citation ────────────────────────────────────────────────────────────

function InlineCitation({ num, source, onOpen }: {
  num:    number;
  source: Source;
  onOpen: (s: Source) => void;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <span className="qa-cite-wrap">
      <button
        className="qa-cite-link"
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        onClick={() => onOpen(source)}
      >
        [{num}]
      </button>
      {hovered && (
        <div className="qa-cite-tooltip">
          <div className="qa-source-tooltip-title">{source.source}{(source.path || source.chapter) ? ` › ${source.path || source.chapter}` : ''}</div>
          <div className="qa-source-tooltip-text">
            {source.text.slice(0, 220)}{source.text.length > 220 ? '…' : ''}
          </div>
        </div>
      )}
    </span>
  );
}

function CitedContent({ content, sources, onOpen }: {
  content: string;
  sources: Source[];
  onOpen:  (s: Source) => void;
}) {
  const parts = content.split(/(\[\d+\])/g);
  return (
    <>
      {parts.map((part, i) => {
        const match = part.match(/^\[(\d+)\]$/);
        if (match) {
          const idx = parseInt(match[1]) - 1;
          const source = sources[idx];
          if (source) {
            return <InlineCitation key={i} num={idx + 1} source={source} onOpen={onOpen} />;
          }
        }
        return <span key={i}>{part}</span>;
      })}
    </>
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

// ── Right panel: branch tree + notes ─────────────────────────────────────────

function RightPanel({
  nodes, rootIds, activeCurrentId,
  onNavigate,
  notes, onDeleteNote, onRefreshNotes, onIndexNote,
}: {
  nodes:           Record<string, ConvNode>;
  rootIds:         string[];
  activeCurrentId: string | null;
  onNavigate:      (id: string) => void;
  notes:           Note[];
  onDeleteNote:    (note_id: string) => void;
  onRefreshNotes:  () => void;
  onIndexNote:     (note_id: string) => Promise<void>;
}) {
  const [tab, setTab] = useState<'tree' | 'notes'>('tree');
  const [expandedNote, setExpandedNote]           = useState<string | null>(null);
  const [expandedContent, setExpandedContent]     = useState('');
  const [expandedQuestions, setExpandedQuestions] = useState<string[]>([]);
  const [indexing, setIndexing]       = useState(false);
  const [indexStatus, setIndexStatus] = useState<'idle' | 'ok' | 'error'>('idle');

  // ── Notes handlers ──
  const handleNoteExpand = async (note_id: string) => {
    try {
      const res  = await fetch(`${API_BASE}/notes/${note_id}`);
      const data = await res.json();
      setExpandedContent(data.content ?? '');
      setExpandedQuestions(data.questions ?? []);
      setExpandedNote(note_id);
      setIndexStatus('idle');
    } catch { /* ignore */ }
  };

  const handleNoteIndex = async () => {
    if (!expandedNote || indexing) return;
    setIndexing(true);
    setIndexStatus('idle');
    try {
      await onIndexNote(expandedNote);
      setIndexStatus('ok');
      setTimeout(() => setIndexStatus('idle'), 2500);
    } catch {
      setIndexStatus('error');
      setTimeout(() => setIndexStatus('idle'), 2500);
    } finally {
      setIndexing(false);
    }
  };

  const handleNoteDelete = async (note_id: string) => {
    if (!confirm('确认删除这条笔记？')) return;
    await fetch(`${API_BASE}/notes/${note_id}`, { method: 'DELETE' });
    onDeleteNote(note_id);
    if (expandedNote === note_id) setExpandedNote(null);
  };

  const expandedNoteObj = notes.find(n => n.note_id === expandedNote);

  return (
    <aside className="kqa-right">
      {/* Panel tab bar */}
      <div className="kqa-rpanel-tabs">
        <button
          className={`kqa-rpanel-tab${tab === 'tree' ? ' kqa-rpanel-tab--active' : ''}`}
          onClick={() => setTab('tree')}
        >对话树</button>
        <button
          className={`kqa-rpanel-tab${tab === 'notes' ? ' kqa-rpanel-tab--active' : ''}`}
          onClick={() => setTab('notes')}
        >知识笔记</button>
      </div>

      {tab === 'tree' ? (
        /* ── Branch map (SVG mind-map) ── */
        <div className="branch-map-scroll">
          <BranchMap
            nodes={nodes}
            rootIds={rootIds}
            activeCurrentId={activeCurrentId}
            onNavigate={onNavigate}
          />
        </div>
      ) : (
        /* ── Notes view ── */
        expandedNote && expandedNoteObj ? (
          <div className="notes-expanded">
            <div className="note-expanded-header">
              <button className="btn btn--icon btn--sm" onClick={() => setExpandedNote(null)}>← 返回</button>
              <div className="note-expanded-title">
                {expandedNoteObj.title}
                {expandedNoteObj.indexed && <span className="note-indexed-badge" title="已加入知识库">知识库</span>}
              </div>
              <div className="note-expanded-meta">{fmtDate(expandedNoteObj.created_at)}</div>
            </div>
            <div className="note-expanded-content note-expanded-content--md">
              <ReactMarkdown>{expandedContent}</ReactMarkdown>
            </div>
            {expandedQuestions.length > 0 && (
              <div className="note-questions">
                <div className="note-questions-label">相关问题</div>
                {expandedQuestions.map((q, i) => (
                  <div key={i} className="note-question-item">
                    <span className="note-question-num">Q{i + 1}</span>{q}
                  </div>
                ))}
              </div>
            )}
            <div className="note-expanded-footer">
              <button
                className={`btn btn--sm ${indexStatus === 'ok' ? 'btn--success' : indexStatus === 'error' ? 'btn--danger' : 'btn--primary'}`}
                onClick={handleNoteIndex}
                disabled={indexing || indexStatus === 'ok'}
                title="将笔记向量化加入知识库，之后问答时会参考此笔记"
              >
                {indexing ? '向量化中…' : indexStatus === 'ok' ? '✓ 已加入' : indexStatus === 'error' ? '失败，重试' : '加入知识库'}
              </button>
              <button className="btn btn--icon btn--sm note-delete-btn" onClick={() => handleNoteDelete(expandedNote)}>删除</button>
            </div>
          </div>
        ) : (
          <>
            <div className="kqa-panel-header">
              <span className="kqa-panel-title">知识笔记</span>
              <button className="btn btn--icon btn--sm" onClick={onRefreshNotes} title="刷新">↻</button>
            </div>
            <div className="notes-list">
              {notes.length === 0 && <p className="kqa-empty">暂无笔记，点击 ✦ 总结对话</p>}
              {notes.map(n => (
                <div key={n.note_id} className="note-item">
                  <div className="note-item-header">
                    <button className="note-item-title" onClick={() => !n.indexing && handleNoteExpand(n.note_id)} disabled={n.indexing}>
                      {n.title}
                    </button>
                    <div className="note-item-actions">
                      {n.indexed && <span className="note-indexed-badge" title="已加入知识库">知识库</span>}
                      <button className="btn btn--icon btn--sm note-delete-btn" onClick={() => handleNoteDelete(n.note_id)} disabled={n.indexing}>✕</button>
                    </div>
                  </div>
                  <div className="note-item-meta">
                    {n.indexing ? <span className="note-indexing">保存中…</span> : fmtDate(n.created_at)}
                  </div>
                </div>
              ))}
            </div>
          </>
        )
      )}
    </aside>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────

export default function KnowledgeQAPage() {
  // ── Session view state ───────────────────────────────────────────────────────
  const [view, setView]           = useState<'list' | 'chat'>('list');
  const [sessionId, setSessionId] = useState('');
  const sessionIdRef              = useRef('');
  useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);

  // ── Tab + conversation tree state ───────────────────────────────────────────
  const tabCountRef               = useRef(1);
  const [tabs, setTabs]           = useState<ConvTab[]>([{ id: 'tab-1', label: '对话 1', currentId: null }]);
  const [activeTabId, setActiveTabId] = useState('tab-1');
  const [nodes, setNodes]         = useState<Record<string, ConvNode>>({});
  const [rootIds, setRootIds]     = useState<string[]>([]);
  const nodesRef                  = useRef<Record<string, ConvNode>>({});
  const rootIdsRef                = useRef<string[]>([]);
  const tabsRef                   = useRef<ConvTab[]>([{ id: 'tab-1', label: '对话 1', currentId: null }]);
  const currentIdRef              = useRef<string | null>(null);
  useEffect(() => { nodesRef.current = nodes; }, [nodes]);
  useEffect(() => { rootIdsRef.current = rootIds; }, [rootIds]);
  useEffect(() => { tabsRef.current = tabs; }, [tabs]);

  const currentId = useMemo(
    () => tabs.find(t => t.id === activeTabId)?.currentId ?? null,
    [tabs, activeTabId],
  );
  useEffect(() => { currentIdRef.current = currentId; }, [currentId]);

  // Derived: path from root to currentId
  const pathNodes = useMemo(() => getPath(nodes, currentId), [nodes, currentId]);

  // ── Other state ─────────────────────────────────────────────────────────────
  const [input, setInput]             = useState('');
  // Per-node loading: tracks nodeIds with in-flight requests so branches don't block each other
  const [loadingNodes, setLoadingNodes] = useState<Set<string>>(new Set());
  const [viewer, setViewer]           = useState<Source | null>(null);
  const [showFullGraph, setShowFullGraph] = useState(false);
  const [summarizing, setSummarizing] = useState(false);
  const [notes, setNotes]             = useState<Note[]>([]);
  const bottomRef   = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const atBottomRef = useRef(true);

  const handleScroll = useCallback(() => {
    const el = messagesRef.current;
    if (!el) return;
    atBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }, []);

  useEffect(() => {
    if (atBottomRef.current) bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [pathNodes]);

  // Navigate: update the active tab's currentId
  const handleSelectNode = useCallback((id: string) => {
    setTabs(prev => prev.map(t => t.id === activeTabId ? { ...t, currentId: id } : t));
    currentIdRef.current = id;
    atBottomRef.current = true;
  }, [activeTabId]);

  // Switch to an existing tab
  const switchTab = useCallback((tabId: string) => {
    const t = tabs.find(tab => tab.id === tabId);
    setActiveTabId(tabId);
    currentIdRef.current = t?.currentId ?? null;
    atBottomRef.current = true;
  }, [tabs]);

  // Close a tab (not the last one)
  const closeTab = useCallback((tabId: string) => {
    setTabs(prev => {
      if (prev.length <= 1) return prev;
      const idx  = prev.findIndex(t => t.id === tabId);
      const next = prev.filter(t => t.id !== tabId);
      if (tabId === activeTabId) {
        const fallback = next[Math.max(0, idx - 1)];
        setActiveTabId(fallback.id);
        currentIdRef.current = fallback.currentId;
      }
      return next;
    });
  }, [activeTabId]);

  // Fork: create a new tab branching from the current position
  const forkFromNode = useCallback(() => {
    const fromId = currentIdRef.current;
    const n = fromId ? nodesRef.current[fromId] : null;
    const rawLabel = n?.question.content ?? '';
    const label = rawLabel.length > 14 ? rawLabel.slice(0, 14) + '…' : rawLabel || `对话 ${tabCountRef.current + 1}`;
    tabCountRef.current += 1;
    const newTab: ConvTab = { id: `tab-${Date.now()}`, label, currentId: fromId };
    setTabs(prev => [...prev, newTab]);
    setActiveTabId(newTab.id);
    currentIdRef.current = fromId;
    atBottomRef.current = true;
  }, []);

  const fetchNotes = useCallback(async () => {
    try {
      const res  = await fetch(`${API_BASE}/notes/list`);
      const data = await res.json();
      setNotes(data.notes ?? []);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => { fetchNotes(); }, [fetchNotes]);

  // ── Submit: create a new ConvNode as child of current node ──────────────────
  const submit = async () => {
    const question = input.trim();
    const isCurrentLoading = !!currentIdRef.current && loadingNodes.has(currentIdRef.current);
    if (!question || isCurrentLoading) return;
    setInput('');
    const tabId = activeTabId;

    // History = current branch path (for LLM context), capped at 12 messages
    const currentPath = getPath(nodesRef.current, currentIdRef.current);
    const history = currentPath.flatMap(n => [
      { role: 'user',      content: n.question.content },
      { role: 'assistant', content: n.answer.content   },
    ]).slice(-12);

    // Create the new node
    const nodeId   = `n-${Date.now()}`;
    const parentId = currentIdRef.current;
    const qMsg: QAMessage = { id: `u-${nodeId}`, role: 'user',      content: question };
    const aMsg: QAMessage = { id: `a-${nodeId}`, role: 'assistant', content: '', sources: [] };
    const newNode: ConvNode = { id: nodeId, parentId, question: qMsg, answer: aMsg, childIds: [] };

    setNodes(prev => {
      const next = { ...prev, [nodeId]: newNode };
      if (parentId && next[parentId]) {
        next[parentId] = { ...next[parentId], childIds: [...next[parentId].childIds, nodeId] };
      }
      nodesRef.current = next;
      return next;
    });
    if (!parentId) setRootIds(prev => [...prev, nodeId]);
    setTabs(prev => prev.map(t => t.id === tabId ? { ...t, currentId: nodeId } : t));
    currentIdRef.current = nodeId;
    atBottomRef.current = true;
    setLoadingNodes(prev => { const s = new Set(prev); s.add(nodeId); return s; });

    // Stream answer
    try {
      const res = await fetch(`${API_BASE}/qa/stream`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message: question, history }),
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
          const raw = line.slice(6).trim();
          if (raw === '[DONE]') continue;
          try {
            const payload = JSON.parse(raw);
            if (payload.sources) {
              setNodes(prev => {
                const next = { ...prev, [nodeId]: { ...prev[nodeId],
                  answer: { ...prev[nodeId].answer, sources: payload.sources, rewritten_query: payload.rewritten_query },
                }};
                nodesRef.current = next;
                return next;
              });
            } else if (payload.graph) {
              setNodes(prev => {
                const next = { ...prev, [nodeId]: { ...prev[nodeId],
                  answer: { ...prev[nodeId].answer, graph: payload.graph },
                }};
                nodesRef.current = next;
                return next;
              });
            } else if (payload.text) {
              fullText += payload.text;
              setNodes(prev => {
                const next = { ...prev, [nodeId]: { ...prev[nodeId],
                  answer: { ...prev[nodeId].answer, content: fullText },
                }};
                nodesRef.current = next;
                return next;
              });
            }
          } catch { /* ignore */ }
        }
      }
    } catch {
      setNodes(prev => {
        const next = { ...prev, [nodeId]: { ...prev[nodeId],
          answer: { ...prev[nodeId].answer, content: '请求失败，请检查后端是否运行' },
        }};
        nodesRef.current = next;
        return next;
      });
    } finally {
      setLoadingNodes(prev => { const s = new Set(prev); s.delete(nodeId); return s; });
    }
  };

  // ── Summarize: root-to-currentId path ───────────────────────────────────────
  const handleSummarize = () => {
    const path = getPath(nodesRef.current, currentIdRef.current);
    const msgs = path.flatMap(n => [
      n.question.content ? { role: 'user',      content: n.question.content } : null,
      n.answer.content   ? { role: 'assistant', content: n.answer.content   } : null,
    ]).filter(Boolean) as { role: string; content: string }[];
    if (msgs.length === 0 || summarizing) return;
    setSummarizing(true);

    const tempId = `temp_${Date.now()}`;
    setNotes(prev => [{
      note_id:    tempId,
      title:      'AI 总结中…',
      size:       0,
      created_at: tempId.slice(5),
      indexing:   true,
    }, ...prev]);

    (async () => {
      try {
        const sumRes = await fetch(`${API_BASE}/qa/summarize`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ messages: msgs }),
        });
        const { title, content, questions = [] } = await sumRes.json();
        const saveRes = await fetch(`${API_BASE}/notes/save`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ title, content, questions }),
        });
        const data = await saveRes.json();
        setNotes(prev => prev.map(n =>
          n.note_id === tempId
            ? { note_id: data.note_id, title, size: data.size, created_at: data.created_at }
            : n
        ));
      } catch {
        setNotes(prev => prev.filter(n => n.note_id !== tempId));
      } finally {
        setSummarizing(false);
      }
    })();
  };

  const handleIndexNote = async (note_id: string) => {
    const res = await fetch(`${API_BASE}/notes/${note_id}/index`, { method: 'POST' });
    if (!res.ok) throw new Error('index failed');
    setNotes((prev) => prev.map((n) => n.note_id === note_id ? { ...n, indexed: true } : n));
  };

  // ── Auto-save session when nodes settle (no in-flight requests) ──────────────
  useEffect(() => {
    if (!sessionId || rootIds.length === 0 || loadingNodes.size > 0) return;
    const title = nodes[rootIds[0]]?.question.content.slice(0, 50) ?? '新对话';
    fetch(`${API_BASE}/qa-sessions/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, title, nodes, root_ids: rootIds, tabs }),
    }).catch(() => {});
  }, [nodes, rootIds, tabs, sessionId, loadingNodes]);

  // ── Session list handlers ────────────────────────────────────────────────────
  const handleLoadSession = async (sid: string) => {
    const res  = await fetch(`${API_BASE}/qa-sessions/${sid}`);
    const data = await res.json();
    const loadedNodes: Record<string, ConvNode> = data.nodes ?? {};
    const loadedRoots: string[]   = data.root_ids ?? [];
    const loadedTabs:  ConvTab[]  = data.tabs ?? [{ id: 'tab-1', label: '对话 1', currentId: null }];
    setNodes(loadedNodes);
    nodesRef.current  = loadedNodes;
    setRootIds(loadedRoots);
    rootIdsRef.current = loadedRoots;
    setTabs(loadedTabs);
    tabsRef.current = loadedTabs;
    // restore tabCountRef from highest tab number
    const maxNum = loadedTabs.reduce((m, t) => {
      const n = parseInt(t.id.replace('tab-', '')) || 0;
      return Math.max(m, n);
    }, 0);
    tabCountRef.current = maxNum || loadedTabs.length;
    setActiveTabId(loadedTabs[0]?.id ?? 'tab-1');
    setSessionId(sid);
    sessionIdRef.current = sid;
    setView('chat');
  };

  const handleNewSession = () => {
    const sid    = crypto.randomUUID();
    const initTab: ConvTab = { id: 'tab-1', label: '对话 1', currentId: null };
    setSessionId(sid);
    sessionIdRef.current = sid;
    setNodes({});
    nodesRef.current = {};
    setRootIds([]);
    rootIdsRef.current = [];
    tabCountRef.current = 1;
    setTabs([initTab]);
    tabsRef.current = [initTab];
    setActiveTabId('tab-1');
    setInput('');
    setView('chat');
  };

  if (view === 'list') {
    return (
      <div className="kqa-page kqa-page--list">
        <SessionListView onLoad={handleLoadSession} onNew={handleNewSession} />
      </div>
    );
  }

  return (
    <div className="kqa-page">
      <KnowledgeSidebar indexedNotes={notes.filter(n => n.indexed && !n.indexing)} />

      <div className="qa-main">
        {/* Session header */}
        <div className="qa-session-bar">
          <button className="btn btn--ghost btn--sm" onClick={() => setView('list')}>← 列表</button>
          <span className="qa-session-bar-title">
            {nodes[rootIds[0]]?.question.content.slice(0, 40) || '新对话'}
          </span>
        </div>

        {/* Tab bar — tabs created via fork, no manual + button */}
        {tabs.length > 1 && (
          <div className="qa-window-tabs">
            {tabs.map(tab => (
              <button
                key={tab.id}
                className={`qa-window-tab${tab.id === activeTabId ? ' qa-window-tab--active' : ''}`}
                onClick={() => switchTab(tab.id)}
              >
                <span className="qa-window-tab-label">{tab.label}</span>
                <span className="qa-window-tab-close" onClick={e => { e.stopPropagation(); closeTab(tab.id); }}>✕</span>
              </button>
            ))}
          </div>
        )}

        <div className="qa-messages" ref={messagesRef} onScroll={handleScroll}>
          {pathNodes.length === 0 && (
            <div className="qa-empty">
              <p className="qa-empty-title">知识库问答</p>
              <p className="qa-empty-hint">问题会检索知识库，将相关内容送入 AI 上下文后回答</p>
            </div>
          )}
          {pathNodes.map(node => {
            const isStreaming = loadingNodes.has(node.id);
            const ans = node.answer;
            return (
              <Fragment key={node.id}>
                {/* Question bubble */}
                <div className="qa-msg qa-msg--user">
                  <div className="qa-msg-label">你</div>
                  <div className="qa-msg-bubble">{node.question.content}</div>
                </div>
                {/* Answer bubble */}
                <div className="qa-msg qa-msg--assistant">
                  <div className="qa-msg-label">AI</div>
                  {ans.rewritten_query && (
                    <div className="qa-sources">
                      <span className="qa-sources-label">意图补全</span>
                      <span className="qa-rewritten-chip">🔍 {ans.rewritten_query}</span>
                    </div>
                  )}
                  <div className="qa-msg-bubble">
                    {ans.sources?.length && ans.content
                      ? <CitedContent content={ans.content} sources={ans.sources} onOpen={setViewer} />
                      : (ans.content || (isStreaming ? <span className="stream-cursor" /> : null))}
                  </div>
                  {ans.sources && ans.sources.length > 0 && (() => {
                    const listSources = ans.sources.filter(s => !s.via_graph);
                    return (
                      <>
                        {listSources.length > 0 && (
                          <div className="qa-sources">
                            <span className="qa-sources-label">参考资料</span>
                            {listSources.map((s, i) => (
                              <SourceChip key={s.chunk_id} index={i} source={s} onOpen={setViewer} />
                            ))}
                          </div>
                        )}
                        {ans.graph && ans.graph.nodes.length > 0 && (
                          <GraphPanel data={ans.graph} sources={ans.sources!} onOpen={setViewer} />
                        )}
                      </>
                    );
                  })()}
                </div>
              </Fragment>
            );
          })}
          <div ref={bottomRef} />
        </div>

        <div className="qa-input-bar">
          <input
            className="qa-input"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); } }}
            placeholder={currentId ? '继续提问，或在右侧树中选择节点…' : '输入问题…'}
            disabled={!!currentId && loadingNodes.has(currentId)}
          />
          <button
            className="btn btn--icon"
            onClick={forkFromNode}
            disabled={!currentId}
            title="从当前位置分叉出新对话标签"
          ><BranchIcon /></button>
          <button className="btn btn--icon" onClick={() => setShowFullGraph(true)} title="查看知识图谱">⬡</button>
          <button
            className="btn btn--icon"
            onClick={handleSummarize}
            disabled={summarizing || pathNodes.length === 0}
            title="总结当前分支对话为知识笔记"
          >{summarizing ? '…' : '✦'}</button>
          <button
            className="btn btn--primary"
            onClick={submit}
            disabled={(!!currentId && loadingNodes.has(currentId)) || !input.trim()}
          >{!!currentId && loadingNodes.has(currentId) ? '…' : '发送'}</button>
        </div>
      </div>

      <RightPanel
        nodes={nodes}
        rootIds={rootIds}
        activeCurrentId={currentId}
        onNavigate={handleSelectNode}
        notes={notes}
        onDeleteNote={id => setNotes(prev => prev.filter(n => n.note_id !== id))}
        onRefreshNotes={fetchNotes}
        onIndexNote={handleIndexNote}
      />

      {viewer && <SourceModal source={viewer} onClose={() => setViewer(null)} />}
      {showFullGraph && <FullGraphViewer onClose={() => setShowFullGraph(false)} />}
    </div>
  );
}
