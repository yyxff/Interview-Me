import { useEffect, useRef } from 'react';
import { Message, InterviewState } from '../types';

interface Props {
  messages: Message[];
  interimTranscript: string;
  state: InterviewState;
}

export function ChatMessages({ messages, interimTranscript, state }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, interimTranscript, state]);

  return (
    <div className="chat-messages">
      {messages.map((msg, i) => {
        const isStreaming = state === 'processing'
          && msg.role === 'assistant'
          && i === messages.length - 1;
        return (
          <div key={msg.id} className={`message message--${msg.role}`}>
            <div className="message-label">
              {msg.role === 'user' ? '你' : '面试官'}
            </div>
            <div className="message-bubble">
              {msg.content}
              {isStreaming && <span className="stream-cursor" />}
            </div>
          </div>
        );
      })}

      {/* 实时显示用户正在说的话 */}
      {interimTranscript && (
        <div className="message message--user message--interim">
          <div className="message-label">你</div>
          <div className="message-bubble">{interimTranscript}</div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
