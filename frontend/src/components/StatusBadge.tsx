import { InterviewState } from '../types';

const STATE_CONFIG: Record<InterviewState, { label: string; color: string; pulse: boolean }> = {
  idle:       { label: '待机',   color: '#6b7280', pulse: false },
  listening:  { label: '聆听中', color: '#22c55e', pulse: true  },
  processing: { label: '思考中', color: '#f59e0b', pulse: true  },
  speaking:   { label: '回答中', color: '#3b82f6', pulse: false },
};

export function StatusBadge({ state }: { state: InterviewState }) {
  const { label, color, pulse } = STATE_CONFIG[state];
  return (
    <div className="status-badge" style={{ borderColor: color, color }}>
      <span className={`status-dot ${pulse ? 'status-dot--pulse' : ''}`} style={{ background: color }} />
      <span>{label}</span>
    </div>
  );
}
