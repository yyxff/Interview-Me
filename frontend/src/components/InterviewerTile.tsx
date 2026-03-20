import { InterviewState } from '../types';

interface Props {
  state: InterviewState;
}

const STATE_LABEL: Record<InterviewState, string | null> = {
  idle:       null,
  listening:  null,
  processing: '思考中…',
  speaking:   '说话中',
};

export function InterviewerTile({ state }: Props) {
  const isSpeaking   = state === 'speaking';
  const isProcessing = state === 'processing';
  const label = STATE_LABEL[state];

  return (
    <div className={`video-tile interviewer-tile ${isSpeaking ? 'tile--speaking' : ''}`}>
      {/* 头像区域 */}
      <div className="interviewer-avatar">
        <div className="avatar-ring-wrapper">
          {isSpeaking && [0, 1, 2].map((i) => (
            <span key={i} className="ripple" style={{ animationDelay: `${i * 0.6}s` }} />
          ))}
          <div className="avatar-ring">
            <svg viewBox="0 0 80 80" fill="none" className="avatar-svg">
              <circle cx="40" cy="32" r="18" fill="#4f46e5" opacity="0.9" />
              <ellipse cx="40" cy="68" rx="26" ry="16" fill="#4f46e5" opacity="0.7" />
            </svg>
          </div>
        </div>

        {/* 说话波形 */}
        {isSpeaking && (
          <div className="wave-bars">
            {[0, 1, 2, 3, 4].map((i) => (
              <span key={i} className="wave-bar" style={{ animationDelay: `${i * 0.12}s` }} />
            ))}
          </div>
        )}

        {/* 处理中转圈 */}
        {isProcessing && <div className="processing-ring" />}
      </div>

      {/* 底部信息栏 */}
      <div className="tile-footer">
        <span className="tile-name">面试官（AI）</span>
        {label && <span className="tile-status-badge">{label}</span>}
      </div>
    </div>
  );
}
