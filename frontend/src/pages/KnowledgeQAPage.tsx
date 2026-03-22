import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
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

function ConvTree({ nodes, rootIds, currentId, loading, onSelect, onNewRoot }: {
  nodes:     Record<string, ConvNode>;
  rootIds:   string[];
  currentId: string | null;
  loading:   boolean;
  onSelect:  (id: string) => void;
  onNewRoot: () => void;
}) {
  const [open, setOpen] = useState(true);

  const activePath = useMemo(() => {
    const set = new Set<string>();
    let cur: string | null = currentId;
    while (cur) { set.add(cur); cur = nodes[cur]?.parentId ?? null; }
    return set;
  }, [nodes, currentId]);

  const renderNode = (id: string, depth = 0): React.ReactNode => {
    const n = nodes[id];
    if (!n) return null;
    const isOnPath = activePath.has(id);
    const isCurrent = id === currentId;
    const label = n.question.content.length > 22
      ? n.question.content.slice(0, 22) + '…'
      : n.question.content;
    return (
      <div key={id}>
        <div
          className={`ctree-node${isOnPath ? ' ctree-node--path' : ''}${isCurrent ? ' ctree-node--current' : ''}`}
          style={{ paddingLeft: 6 + depth * 12 }}
          onClick={() => !loading && onSelect(id)}
          title={n.question.content}
        >
          <span className="ctree-dot" />
          <span className="ctree-label">{label}</span>
          {n.childIds.length > 1 && (
            <span className="ctree-fork-badge">{n.childIds.length}</span>
          )}
        </div>
        {n.childIds.map(cid => renderNode(cid, depth + 1))}
      </div>
    );
  };

  if (rootIds.length === 0) return null;

  return (
    <div className="ctree-panel">
      <div className="ctree-header">
        <span className="ctree-title" onClick={() => setOpen(o => !o)}>
          {open ? '▾' : '▸'} 对话树
          <span className="ctree-count">{Object.keys(nodes).length}</span>
        </span>
        <button className="ctree-new-btn" title="新对话（从根开始）" onClick={onNewRoot}>＋</button>
      </div>
      {open && (
        <div className="ctree-body">
          {rootIds.map(id => renderNode(id))}
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

// ── Notes sidebar (right) ─────────────────────────────────────────────────────

function NotesSidebar({ notes, onDelete, onRefresh, onIndex }: {
  notes:     Note[];
  onDelete:  (note_id: string) => void;
  onRefresh: () => void;
  onIndex:   (note_id: string) => Promise<void>;
}) {
  const [expanded, setExpanded]               = useState<string | null>(null);
  const [expandedContent, setExpandedContent] = useState('');
  const [expandedQuestions, setExpandedQuestions] = useState<string[]>([]);
  const [indexing, setIndexing]               = useState(false);
  const [indexStatus, setIndexStatus]         = useState<'idle' | 'ok' | 'error'>('idle');

  const handleExpand = async (note_id: string) => {
    try {
      const res  = await fetch(`${API_BASE}/notes/${note_id}`);
      const data = await res.json();
      setExpandedContent(data.content ?? '');
      setExpandedQuestions(data.questions ?? []);
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
        <div className="note-expanded-content note-expanded-content--md">
          <ReactMarkdown>{expandedContent}</ReactMarkdown>
        </div>
        {expandedQuestions.length > 0 && (
          <div className="note-questions">
            <div className="note-questions-label">相关问题</div>
            {expandedQuestions.map((q, i) => (
              <div key={i} className="note-question-item">
                <span className="note-question-num">Q{i + 1}</span>
                {q}
              </div>
            ))}
          </div>
        )}
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
  // ── Forked conversation tree state ──────────────────────────────────────────
  const [nodes, setNodes]         = useState<Record<string, ConvNode>>({});
  const [rootIds, setRootIds]     = useState<string[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const nodesRef    = useRef<Record<string, ConvNode>>({});
  const currentIdRef = useRef<string | null>(null);
  useEffect(() => { nodesRef.current = nodes; }, [nodes]);

  // Derived: messages to display = path from root to currentId
  const pathMessages = useMemo(() => {
    const path = getPath(nodes, currentId);
    return path.flatMap(n => [n.question, n.answer]);
  }, [nodes, currentId]);

  // ── Other state ─────────────────────────────────────────────────────────────
  const [input, setInput]             = useState('');
  const [loading, setLoading]         = useState(false);
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
  }, [pathMessages]);

  // Scroll to bottom whenever we switch branches
  const handleSelectNode = useCallback((id: string) => {
    setCurrentId(id);
    currentIdRef.current = id;
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

  // ── Submit: create a new ConvNode as child of currentId ─────────────────────
  const submit = async () => {
    const question = input.trim();
    if (!question || loading) return;
    setInput('');

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
    setCurrentId(nodeId);
    currentIdRef.current = nodeId;
    atBottomRef.current = true;
    setLoading(true);

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
      setLoading(false);
    }
  };

  // ── Summarize: only current branch path ─────────────────────────────────────
  const handleSummarize = () => {
    const path = getPath(nodesRef.current, currentIdRef.current);
    const msgs = path.flatMap(n => [
      n.question.content ? { role: 'user',      content: n.question.content } : null,
      n.answer.content   ? { role: 'assistant', content: n.answer.content   } : null,
    ]).filter(Boolean) as { role: string; content: string }[];

    if (msgs.length === 0 || summarizing) return;
    setSummarizing(true);

    const tempId = `temp_${Date.now()}`;
    setNotes((prev) => [{
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
    setNotes((prev) => prev.map((n) => n.note_id === note_id ? { ...n, indexed: true } : n));
  };

  const hasBranch = currentId != null && (nodesRef.current[currentId]?.childIds.length ?? 0) > 0;

  return (
    <div className="kqa-page">
      <KnowledgeSidebar indexedNotes={notes.filter((n) => n.indexed && !n.indexing)} />

      <div className="qa-main">
        {/* Conversation tree — floating panel top-right */}
        <ConvTree
          nodes={nodes}
          rootIds={rootIds}
          currentId={currentId}
          loading={loading}
          onSelect={handleSelectNode}
          onNewRoot={() => { setCurrentId(null); currentIdRef.current = null; }}
        />

        <div className="qa-messages" ref={messagesRef} onScroll={handleScroll}>
          {pathMessages.length === 0 && (
            <div className="qa-empty">
              <p className="qa-empty-title">知识库问答</p>
              <p className="qa-empty-hint">
                {rootIds.length > 0
                  ? '点击左侧树节点切换分支，或在下方输入新问题开始新对话'
                  : '问题会检索知识库，将相关内容送入 AI 上下文后回答'}
              </p>
            </div>
          )}
          {pathMessages.map((msg) => (
            <div key={msg.id} className={`qa-msg qa-msg--${msg.role}`}>
              <div className="qa-msg-label">{msg.role === 'user' ? '你' : 'AI'}</div>
              {msg.role === 'assistant' && msg.rewritten_query && (
                <div className="qa-sources">
                  <span className="qa-sources-label">意图补全</span>
                  <span className="qa-rewritten-chip">🔍 {msg.rewritten_query}</span>
                </div>
              )}
              <div className="qa-msg-bubble">
                {msg.role === 'assistant' && msg.sources?.length && msg.content
                  ? <CitedContent content={msg.content} sources={msg.sources} onOpen={setViewer} />
                  : (msg.content || (loading && msg.role === 'assistant'
                      ? <span className="stream-cursor" /> : null))}
              </div>
              {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (() => {
                const listSources = msg.sources.filter(s => !s.via_graph);
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
                    {msg.graph && msg.graph.nodes.length > 0 && (
                      <GraphPanel data={msg.graph} sources={msg.sources!} onOpen={setViewer} />
                    )}
                  </>
                );
              })()}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        <div className="qa-input-bar">
          {hasBranch && (
            <span className="qa-branch-tip" title="当前节点已有子分支，继续输入将创建新分支">
              ⑂ 分叉
            </span>
          )}
          <input
            className="qa-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); } }}
            placeholder={currentId ? '继续提问，或在树中选择其他节点切换分支…' : '输入问题…'}
            disabled={loading}
          />
          <button
            className="btn btn--icon"
            onClick={() => setShowFullGraph(true)}
            title="查看知识图谱"
          >⬡</button>
          <button
            className="btn btn--icon"
            onClick={handleSummarize}
            disabled={summarizing || pathMessages.length === 0}
            title="总结当前分支对话为知识点"
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
      {showFullGraph && <FullGraphViewer onClose={() => setShowFullGraph(false)} />}
    </div>
  );
}
