import { useRef, useCallback, useEffect } from 'react';

interface Options {
  lang?: string;
  /** 检测到语音开始（可用于打断 AI） */
  onSpeechStart?: () => void;
  /** 实时识别的中间结果 */
  onInterimResult?: (transcript: string) => void;
  /** 一句话识别完成的最终结果 */
  onFinalResult?: (transcript: string) => void;
  onError?: (error: string) => void;
}

export function useSpeechRecognition({
  lang = 'zh-CN',
  onSpeechStart,
  onInterimResult,
  onFinalResult,
  onError,
}: Options) {
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const shouldRunRef = useRef(false);
  const langRef = useRef(lang);

  // 始终保持 callbacks 最新，避免 stale closure
  const callbacksRef = useRef({ onSpeechStart, onInterimResult, onFinalResult, onError });
  useEffect(() => {
    callbacksRef.current = { onSpeechStart, onInterimResult, onFinalResult, onError };
  });
  useEffect(() => {
    langRef.current = lang;
  }, [lang]);

  const initRecognition = useCallback(() => {
    const SpeechRecognitionAPI =
      window.SpeechRecognition ||
      (window as unknown as { webkitSpeechRecognition: typeof SpeechRecognition })
        .webkitSpeechRecognition;

    if (!SpeechRecognitionAPI) {
      callbacksRef.current.onError?.('浏览器不支持 SpeechRecognition，请使用 Chrome');
      return;
    }

    const recognition = new SpeechRecognitionAPI();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = langRef.current;
    recognition.maxAlternatives = 1;

    recognition.onspeechstart = () => {
      callbacksRef.current.onSpeechStart?.();
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let interim = '';
      let final = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const r = event.results[i];
        if (r.isFinal) {
          final += r[0].transcript;
        } else {
          interim += r[0].transcript;
        }
      }
      if (interim) callbacksRef.current.onInterimResult?.(interim);
      if (final) callbacksRef.current.onFinalResult?.(final);
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      // no-speech / aborted 不算错误，静默处理
      if (event.error === 'no-speech' || event.error === 'aborted') return;
      callbacksRef.current.onError?.(event.error);
    };

    // Chrome 会在静默一段时间后自动停止，这里自动重启
    recognition.onend = () => {
      if (shouldRunRef.current) {
        setTimeout(() => {
          if (shouldRunRef.current) {
            initRecognition();
          }
        }, 150);
      }
    };

    recognitionRef.current = recognition;
    try {
      recognition.start();
    } catch {
      // 已有实例在运行，忽略
    }
  }, []);

  const start = useCallback(() => {
    if (shouldRunRef.current) return;
    shouldRunRef.current = true;
    initRecognition();
  }, [initRecognition]);

  const stop = useCallback(() => {
    shouldRunRef.current = false;
    recognitionRef.current?.abort();
    recognitionRef.current = null;
  }, []);

  useEffect(() => () => { stop(); }, [stop]);

  return { start, stop };
}
