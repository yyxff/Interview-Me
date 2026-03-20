import { ChatMessages } from './ChatMessages';
import { Message, InterviewState } from '../types';

interface Props {
  messages: Message[];
  interimTranscript: string;
  state: InterviewState;
  onClose: () => void;
}

export function ChatPanel({ messages, interimTranscript, state, onClose }: Props) {
  const isEmpty = messages.length === 0 && !interimTranscript
    && state !== 'processing' && state !== 'speaking';

  return (
    <aside className="chat-panel">
      <div className="chat-panel-header">
        <span className="chat-panel-title">对话记录</span>
        <button className="btn btn--icon" onClick={onClose} title="关闭">
          ✕
        </button>
      </div>
      <div className="chat-panel-body">
        {isEmpty ? (
          <p className="chat-empty">面试开始后，对话记录会显示在这里。</p>
        ) : (
          <ChatMessages messages={messages} interimTranscript={interimTranscript} state={state} />
        )}
      </div>
    </aside>
  );
}
