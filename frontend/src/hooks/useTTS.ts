import { useRef, useCallback, useEffect } from 'react';

interface Options {
  lang?: string;
  rate?: number;
  pitch?: number;
  onStart?: () => void;
  onEnd?: () => void;
}

export function useTTS({ lang = 'zh-CN', rate = 1, pitch = 1, onStart, onEnd }: Options = {}) {
  const callbacksRef = useRef({ onStart, onEnd });
  useEffect(() => {
    callbacksRef.current = { onStart, onEnd };
  });

  // Chrome bug: SpeechSynthesis 在 ~15s 后会静默停止，需要定期 pause/resume 刷新
  useEffect(() => {
    const id = setInterval(() => {
      if (window.speechSynthesis.speaking && !window.speechSynthesis.paused) {
        window.speechSynthesis.pause();
        window.speechSynthesis.resume();
      }
    }, 10_000);
    return () => clearInterval(id);
  }, []);

  const speak = useCallback(
    (text: string) => {
      window.speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = lang;
      utterance.rate = rate;
      utterance.pitch = pitch;

      // 优先选择本地中文语音
      const pickVoice = () => {
        const voices = window.speechSynthesis.getVoices();
        return (
          voices.find((v) => v.lang.startsWith(lang) && v.localService) ||
          voices.find((v) => v.lang.startsWith(lang)) ||
          null
        );
      };

      const voice = pickVoice();
      if (voice) utterance.voice = voice;

      utterance.onstart = () => callbacksRef.current.onStart?.();
      utterance.onend = () => callbacksRef.current.onEnd?.();
      utterance.onerror = () => callbacksRef.current.onEnd?.();

      // 某些浏览器在 getVoices() 尚未加载完时需要等待
      if (window.speechSynthesis.getVoices().length === 0) {
        window.speechSynthesis.onvoiceschanged = () => {
          const v = pickVoice();
          if (v) utterance.voice = v;
          window.speechSynthesis.speak(utterance);
        };
      } else {
        window.speechSynthesis.speak(utterance);
      }
    },
    [lang, rate, pitch],
  );

  const cancel = useCallback(() => {
    window.speechSynthesis.cancel();
  }, []);

  useEffect(() => () => { cancel(); }, [cancel]);

  return { speak, cancel };
}
