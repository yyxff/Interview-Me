import { useState, useCallback, useRef, useEffect } from 'react';
import { Message, InterviewState } from '../types';
import { useSpeechRecognition } from '../hooks/useSpeechRecognition';
import { useTTS } from '../hooks/useTTS';
import { InterviewerTile } from '../components/InterviewerTile';
import { UserTile } from '../components/UserTile';
import { ChatPanel } from '../components/ChatPanel';
import { StatusBadge } from '../components/StatusBadge';

const API_BASE = 'http://localhost:8000';
const SILENCE_TIMEOUT_MS = 800;

export default function InterviewPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [state, setState] = useState<InterviewState>('idle');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [isStarted, setIsStarted] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resumeStatus, setResumeStatus] = useState<'none' | 'uploading' | 'ready'>('none');

  const sessionIdRef = useRef(crypto.randomUUID());
  const stateRef = useRef<InterviewState>('idle');
  const messagesRef = useRef<Message[]>([]);
  const accumulatedRef = useRef('');
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ttsActionsRef = useRef({ speak: (_: string) => {}, cancel: () => {} });
  const recognitionActionsRef = useRef({ start: () => {}, stop: () => {} });

  useEffect(() => { stateRef.current = state; }, [state]);
  useEffect(() => { messagesRef.current = messages; }, [messages]);

  const { speak, cancel: cancelSpeak } = useTTS({
    lang: 'zh-CN',
    rate: 1,
    onStart: () => {
      setState('speaking');
      recognitionActionsRef.current.stop();
    },
    onEnd: () => {
      if (stateRef.current !== 'idle') {
        setState('listening');
        recognitionActionsRef.current.start();
      }
    },
  });

  useEffect(() => {
    ttsActionsRef.current = { speak, cancel: cancelSpeak };
  }, [speak, cancelSpeak]);

  const submitTurn = useCallback(async () => {
    const text = accumulatedRef.current.trim();
    if (!text || stateRef.current === 'processing') return;

    accumulatedRef.current = '';
    setInterimTranscript('');

    const history = messagesRef.current.map((m) => ({ role: m.role, content: m.content }));
    const userMsg: Message = { id: `u-${Date.now()}`, role: 'user', content: text, timestamp: new Date() };
    setMessages((prev) => [...prev, userMsg]);
    setState('processing');

    const assistantId = `a-${Date.now()}`;
    setMessages((prev) => [...prev, { id: assistantId, role: 'assistant' as const, content: '', timestamp: new Date() }]);

    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history, session_id: sessionIdRef.current }),
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
            const { text: chunk, error } = JSON.parse(data);
            if (error) throw new Error(error);
            fullText += chunk;
            setMessages((prev) => prev.map((m) => m.id === assistantId ? { ...m, content: fullText } : m));
          } catch { /* ignore malformed chunk */ }
        }
      }

      if (fullText) ttsActionsRef.current.speak(fullText);
      else setState('listening');
    } catch (err) {
      console.error('Backend error:', err);
      setError('无法连接到后端，请确认 ./start.sh 已运行');
      setState('listening');
    }
  }, []);

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
    silenceTimerRef.current = setTimeout(submitTurn, SILENCE_TIMEOUT_MS);
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

  const handleResumeUpload = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.pdf')) { setError('只支持 PDF 格式的简历'); return; }
    setResumeStatus('uploading');
    const form = new FormData();
    form.append('file', file);
    form.append('session_id', sessionIdRef.current);
    try {
      const res = await fetch(`${API_BASE}/upload/resume`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setResumeStatus('ready');
    } catch {
      setError('简历上传失败，请重试');
      setResumeStatus('none');
    }
  }, []);

  const handleStart = useCallback(() => {
    setError(null);
    setIsStarted(true);
    setState('listening');
    startRecognition();
  }, [startRecognition]);

  const handleStop = useCallback(() => {
    setIsStarted(false);
    setState('idle');
    stopRecognition();
    ttsActionsRef.current.cancel();
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    accumulatedRef.current = '';
    setInterimTranscript('');
  }, [stopRecognition]);

  return (
    <div className="interview-page">
      {/* 顶部状态栏 */}
      <div className="interview-topbar">
        <StatusBadge state={state} />
        <div className="interview-topbar-actions">
          <label
            className={`btn btn--icon ${resumeStatus === 'ready' ? 'btn--active' : ''}`}
            title={resumeStatus === 'ready' ? '简历已上传' : resumeStatus === 'uploading' ? '上传中…' : '上传简历 PDF'}
            style={{ cursor: resumeStatus === 'uploading' ? 'wait' : 'pointer' }}
          >
            {resumeStatus === 'uploading' ? '⏳' : resumeStatus === 'ready' ? '📄' : '📎'}
            <input type="file" accept=".pdf" style={{ display: 'none' }}
              disabled={resumeStatus === 'uploading'}
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleResumeUpload(f); }} />
          </label>
          <button className={`btn btn--icon ${showChat ? 'btn--active' : ''}`}
            onClick={() => setShowChat((v) => !v)} title="对话记录">💬</button>
        </div>
      </div>

      {/* 主区域 */}
      <div className="interview-content">
        <div className="video-area">
          <InterviewerTile state={state} />
          <UserTile state={state} />
          {!isStarted && (
            <div className="start-overlay">
              <p className="overlay-hint">系统自动检测语音 · 停顿 0.8s 发送 · AI 说完后自动开麦</p>
              <button className="btn btn--primary btn--lg" onClick={handleStart}>开始面试</button>
            </div>
          )}
        </div>

        {showChat && (
          <ChatPanel messages={messages} interimTranscript={interimTranscript}
            state={state} onClose={() => setShowChat(false)} />
        )}
      </div>

      {error && (
        <div className="error-bar">
          <span>{error}</span>
          <button className="btn btn--icon btn--sm" onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {isStarted && (
        <footer className="footer">
          <button className="btn btn--danger btn--lg" onClick={handleStop}>结束面试</button>
        </footer>
      )}
    </div>
  );
}
