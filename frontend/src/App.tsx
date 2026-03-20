import { useState, useCallback, useRef, useEffect } from 'react';
import { Message, InterviewState } from './types';
import { useSpeechRecognition } from './hooks/useSpeechRecognition';
import { useTTS } from './hooks/useTTS';
import { InterviewerTile } from './components/InterviewerTile';
import { UserTile } from './components/UserTile';
import { ChatPanel } from './components/ChatPanel';
import { StatusBadge } from './components/StatusBadge';
import './App.css';

const API_BASE = 'http://localhost:8000';
const SILENCE_TIMEOUT_MS = 800;

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [state, setState] = useState<InterviewState>('idle');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [isStarted, setIsStarted] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ---- Refs：在 async/事件回调中安全读取最新状态 ----
  const stateRef = useRef<InterviewState>('idle');
  const messagesRef = useRef<Message[]>([]);
  const accumulatedRef = useRef('');
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ttsActionsRef = useRef({ speak: (_: string) => {}, cancel: () => {} });
  // 桥接：useTTS 回调中需要控制识别，但识别 hook 在后面声明
  const recognitionActionsRef = useRef({ start: () => {}, stop: () => {} });

  useEffect(() => { stateRef.current = state; }, [state]);
  useEffect(() => { messagesRef.current = messages; }, [messages]);

  // ---- TTS ----
  const { speak, cancel: cancelSpeak } = useTTS({
    lang: 'zh-CN',
    rate: 1,
    onStart: () => {
      setState('speaking');
      recognitionActionsRef.current.stop(); // 整个播放期间停麦
    },
    onEnd: () => {
      if (stateRef.current !== 'idle') {
        setState('listening');
        recognitionActionsRef.current.start(); // 播完再开麦
      }
    },
  });

  useEffect(() => {
    ttsActionsRef.current = { speak, cancel: cancelSpeak };
  }, [speak, cancelSpeak]);


  // ---- 向后端发送本轮对话 ----
  const submitTurn = useCallback(async () => {
    const text = accumulatedRef.current.trim();
    if (!text || stateRef.current === 'processing') return;

    accumulatedRef.current = '';
    setInterimTranscript('');

    const history = messagesRef.current.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    const userMsg: Message = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setState('processing');

    // 立即占位，流式内容逐字填入
    const assistantId = `a-${Date.now()}`;
    setMessages((prev) => [...prev, {
      id: assistantId, role: 'assistant' as const, content: '', timestamp: new Date(),
    }]);

    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history }),
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
            setMessages((prev) =>
              prev.map((m) => m.id === assistantId ? { ...m, content: fullText } : m)
            );
          } catch { /* ignore malformed chunk */ }
        }
      }

      if (fullText) {
        ttsActionsRef.current.speak(fullText);
      } else {
        setState('listening');
      }
    } catch (err) {
      console.error('Backend error:', err);
      setError('无法连接到后端，请确认 ./start.sh 已运行');
      setState('listening');
    }
  }, []);

  // ---- 语音识别回调 ----
  const handleSpeechStart = useCallback(() => {
    if (stateRef.current !== 'processing') setState('listening');
  }, []);

  const handleInterimResult = useCallback((transcript: string) => {
    setInterimTranscript(accumulatedRef.current + transcript);
  }, []);

  const handleFinalResult = useCallback(
    (transcript: string) => {
      accumulatedRef.current += transcript;
      setInterimTranscript(accumulatedRef.current);
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = setTimeout(submitTurn, SILENCE_TIMEOUT_MS);
    },
    [submitTurn],
  );

  const { start: startRecognition, stop: stopRecognition } = useSpeechRecognition({
    lang: 'zh-CN',
    onSpeechStart:   handleSpeechStart,
    onInterimResult: handleInterimResult,
    onFinalResult:   handleFinalResult,
    onError: (err) => setError(`语音识别错误: ${err}`),
  });

  useEffect(() => {
    recognitionActionsRef.current = { start: startRecognition, stop: stopRecognition };
  }, [startRecognition, stopRecognition]);

  // ---- 控制 ----
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
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-left">
          <span className="logo">🎯</span>
          <span className="header-title">模拟面试</span>
        </div>
        <div className="header-center">
          <StatusBadge state={state} />
        </div>
        <div className="header-right">
          <button
            className={`btn btn--icon ${showChat ? 'btn--active' : ''}`}
            onClick={() => setShowChat((v) => !v)}
            title="对话记录"
          >
            💬
          </button>
        </div>
      </header>

      {/* ── Main content ── */}
      <div className="content">
        {/* 视频区 */}
        <div className="video-area">
          <InterviewerTile state={state} />
          <UserTile state={state} />

          {/* 未开始时的覆盖层 */}
          {!isStarted && (
            <div className="start-overlay">
              <p className="overlay-hint">
                系统自动检测语音 · 停顿 1.5s 发送 · 开口即可打断 AI
              </p>
              <button className="btn btn--primary btn--lg" onClick={handleStart}>
                开始面试
              </button>
            </div>
          )}
        </div>

        {/* 可选对话面板 */}
        {showChat && (
          <ChatPanel
            messages={messages}
            interimTranscript={interimTranscript}
            state={state}
            onClose={() => setShowChat(false)}
          />
        )}
      </div>

      {/* ── Error bar ── */}
      {error && (
        <div className="error-bar">
          <span>{error}</span>
          <button className="btn btn--icon btn--sm" onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {/* ── Footer ── */}
      {isStarted && (
        <footer className="footer">
          <button className="btn btn--danger btn--lg" onClick={handleStop}>
            结束面试
          </button>
        </footer>
      )}
    </div>
  );
}
