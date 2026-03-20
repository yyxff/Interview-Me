// Web Speech API 类型声明（TypeScript DOM lib 覆盖不完整）
interface SpeechRecognitionResult {
  readonly isFinal: boolean;
  readonly length: number;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
}

interface SpeechRecognitionAlternative {
  readonly transcript: string;
  readonly confidence: number;
}

interface SpeechRecognitionResultList {
  readonly length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionEvent extends Event {
  readonly resultIndex: number;
  readonly results: SpeechRecognitionResultList;
}

interface SpeechRecognitionErrorEvent extends Event {
  readonly error: string;
  readonly message: string;
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  maxAlternatives: number;
  onspeechstart: ((this: SpeechRecognition, ev: Event) => void) | null;
  onspeechend:   ((this: SpeechRecognition, ev: Event) => void) | null;
  onresult:      ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => void) | null;
  onerror:       ((this: SpeechRecognition, ev: SpeechRecognitionErrorEvent) => void) | null;
  onend:         ((this: SpeechRecognition, ev: Event) => void) | null;
  onstart:       ((this: SpeechRecognition, ev: Event) => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}

declare var SpeechRecognition: {
  prototype: SpeechRecognition;
  new (): SpeechRecognition;
};

interface Window {
  SpeechRecognition: typeof SpeechRecognition;
  webkitSpeechRecognition: typeof SpeechRecognition;
}
