import { useState, useEffect, useRef } from 'react';

const API_BASE = 'http://localhost:8000';

interface KnowledgeFile {
  name: string;
  filename: string;
  size: number;
  indexed: boolean;
}

interface ProfileStatus {
  indexed:  boolean;
  chunks:   number;
  sections: string[];
}

interface IndexProgress {
  status: 'idle' | 'running' | 'done' | 'error';
  file: string;
  chunks_done: number;
  chunks_total: number;
  vectors_added: number;
  elapsed_s: number;
  eta_s: number | null;
  error: string | null;
}

function fmtSeconds(s: number | null): string {
  if (s === null || s < 0) return '…';
  if (s < 60) return `${Math.round(s)}s`;
  return `${Math.floor(s / 60)}m${Math.round(s % 60)}s`;
}

export default function KnowledgePage() {
  const [files, setFiles] = useState<KnowledgeFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [mdContent, setMdContent] = useState('');
  const [loadingContent, setLoadingContent] = useState(false);
  const [progress, setProgress] = useState<IndexProgress | null>(null);
  const [profile, setProfile]           = useState<ProfileStatus | null>(null);
  const [profileUploading, setProfileUploading] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);
  const fileInputRef        = useRef<HTMLInputElement>(null);
  const profileInputRef     = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchList = async () => {
    try {
      const res = await fetch(`${API_BASE}/knowledge/list`);
      const data = await res.json();
      setFiles(data.files ?? []);
    } catch { /* ignore */ }
  };

  const fetchProgress = async () => {
    try {
      const res = await fetch(`${API_BASE}/rag/index-progress`);
      const data: IndexProgress = await res.json();
      setProgress(data);
      // 刷新文件列表（索引完成后 indexed 状态会变）
      if (data.status === 'done' || data.status === 'error') {
        fetchList();
      }
    } catch { /* ignore */ }
  };

  // 轮询进度
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
      setProfileError('只支持 Markdown (.md) 文件');
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

  useEffect(() => {
    if (files.length > 0 && !selected) setSelected(files[0].name);
  }, [files]);

  useEffect(() => {
    if (!selected) return;
    setLoadingContent(true);
    setMdContent('');
    fetch(`${API_BASE}/knowledge/${selected}`)
      .then((r) => r.json())
      .then((d) => setMdContent(d.content ?? ''))
      .catch(() => setMdContent('加载失败'))
      .finally(() => setLoadingContent(false));
  }, [selected]);

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
      const data = await res.json();
      await fetchList();
      setSelected(data.name);
    } catch (e: any) {
      setUploadError(e.message ?? '上传失败');
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const isIndexing = progress?.status === 'running';
  const pct = progress && progress.chunks_total > 0
    ? Math.round(progress.chunks_done / progress.chunks_total * 100)
    : 0;

  return (
    <div className="knowledge-page">
      {/* 左侧边栏 */}
      <aside className="knowledge-sidebar">
        <div className="knowledge-sidebar-header">
          <span className="knowledge-sidebar-title">知识库</span>
          <label className={`btn btn--sm btn--primary ${uploading || isIndexing ? 'btn--disabled' : ''}`}
            title="上传 EPUB">
            {uploading ? '处理中…' : '+ 上传'}
            <input ref={fileInputRef} type="file" accept=".epub" style={{ display: 'none' }}
              disabled={uploading || isIndexing}
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleUpload(f); }} />
          </label>
        </div>

        {/* 索引进度条 */}
        {isIndexing && progress && (
          <div className="index-progress">
            <div className="index-progress-bar">
              <div className="index-progress-fill" style={{ width: `${pct}%` }} />
            </div>
            <div className="index-progress-text">
              <span>向量化中 {progress.chunks_done}/{progress.chunks_total} 段</span>
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

        <div className="knowledge-file-list">
          {files.length === 0 && !uploading && (
            <p className="knowledge-empty">还没有文件，上传 EPUB 开始构建知识库。</p>
          )}
          {files.map((f) => (
            <button
              key={f.name}
              className={`knowledge-file-btn ${selected === f.name ? 'knowledge-file-btn--active' : ''}`}
              onClick={() => setSelected(f.name)}
            >
              <span className="knowledge-file-btn-name">{f.name}</span>
              <span className="knowledge-file-btn-meta">
                {(f.size / 1024).toFixed(0)}K
                <span className={`badge ${f.indexed ? 'badge--green' : isIndexing ? 'badge--amber' : 'badge--gray'}`}>
                  {f.indexed ? '已索引' : isIndexing ? '索引中' : '未索引'}
                </span>
              </span>
            </button>
          ))}
        </div>

        {/* Profile 区域 */}
        <div className="profile-section">
          <div className="profile-section-header">
            <span className="profile-section-title">个人 Profile</span>
            <span className={`badge ${profile?.indexed ? 'badge--green' : 'badge--gray'}`}>
              {profile?.indexed ? `${profile.chunks} 段` : '未上传'}
            </span>
            <label className={`btn btn--sm ${profileUploading ? 'btn--disabled' : 'btn--ghost'}`} title="上传 MD 简历">
              {profileUploading ? '上传中…' : '上传'}
              <input ref={profileInputRef} type="file" accept=".md" style={{ display: 'none' }}
                disabled={profileUploading}
                onChange={(e) => { const f = e.target.files?.[0]; if (f) handleProfileUpload(f); }} />
            </label>
          </div>
          {profileError && <p className="knowledge-upload-error">{profileError}</p>}
          {profile?.indexed && profile.sections.length > 0 && (
            <ul className="profile-sections">
              {profile.sections.map(s => <li key={s}>{s}</li>)}
            </ul>
          )}
        </div>
      </aside>

      {/* 右侧内容区 */}
      <main className="knowledge-viewer">
        {!selected ? (
          <div className="knowledge-viewer-empty">选择左侧文件查看内容</div>
        ) : loadingContent ? (
          <div className="knowledge-viewer-empty">加载中…</div>
        ) : (
          <>
            <div className="knowledge-viewer-header">
              <h2 className="knowledge-viewer-title">{selected}</h2>
              <span className="knowledge-viewer-hint">转换后的 Markdown · 选中文字后点击可复制</span>
            </div>
            <pre className="knowledge-viewer-content"
              onClick={() => {
                const sel = window.getSelection()?.toString();
                if (sel) navigator.clipboard?.writeText(sel);
              }}
            >{mdContent}</pre>
          </>
        )}
      </main>
    </div>
  );
}
