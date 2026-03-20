import { useState, useEffect, useRef } from 'react';

const API_BASE = 'http://localhost:8000';

interface KnowledgeFile {
  name: string;
  filename: string;
  size: number;
  indexed: boolean;
}

interface Props {
  onClose: () => void;
}

export function KnowledgePanel({ onClose }: Props) {
  const [files, setFiles] = useState<KnowledgeFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [mdContent, setMdContent] = useState<string>('');
  const [loadingContent, setLoadingContent] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchList = async () => {
    try {
      const res = await fetch(`${API_BASE}/knowledge/list`);
      const data = await res.json();
      setFiles(data.files);
    } catch { /* ignore */ }
  };

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
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      await fetchList();
    } catch (e: any) {
      setUploadError(e.message || '上传失败');
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleViewContent = async (name: string) => {
    if (selectedFile === name) { setSelectedFile(null); setMdContent(''); return; }
    setSelectedFile(name);
    setLoadingContent(true);
    try {
      const res = await fetch(`${API_BASE}/knowledge/${name}`);
      const data = await res.json();
      setMdContent(data.content);
    } catch {
      setMdContent('加载失败');
    } finally {
      setLoadingContent(false);
    }
  };

  return (
    <aside className="knowledge-panel">
      <div className="chat-panel-header">
        <span className="chat-panel-title">📚 知识库管理</span>
        <button className="btn btn--icon" onClick={onClose}>✕</button>
      </div>

      <div className="knowledge-upload-area">
        <label className={`knowledge-upload-btn ${uploading ? 'uploading' : ''}`}>
          {uploading ? '处理中…' : '+ 上传 EPUB'}
          <input
            ref={fileInputRef}
            type="file"
            accept=".epub"
            style={{ display: 'none' }}
            disabled={uploading}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) handleUpload(f); }}
          />
        </label>
        <span className="knowledge-upload-hint">EPUB → Markdown → 自动索引</span>
        {uploadError && <p className="knowledge-error">{uploadError}</p>}
      </div>

      <div className="knowledge-file-list">
        {files.length === 0 && (
          <p className="chat-empty">还没有知识库文件，上传 EPUB 开始构建。</p>
        )}
        {files.map((f) => (
          <div key={f.name} className="knowledge-file-item">
            <div className="knowledge-file-info" onClick={() => handleViewContent(f.name)}>
              <span className="knowledge-file-name">{f.name}</span>
              <span className="knowledge-file-meta">
                {(f.size / 1024).toFixed(0)} KB
                {f.indexed
                  ? <span className="badge badge--green">已索引</span>
                  : <span className="badge badge--gray">未索引</span>}
              </span>
            </div>

            {selectedFile === f.name && (
              <div className="knowledge-md-viewer">
                {loadingContent
                  ? <p className="chat-empty">加载中…</p>
                  : <pre className="knowledge-md-content">{mdContent}</pre>}
              </div>
            )}
          </div>
        ))}
      </div>
    </aside>
  );
}
