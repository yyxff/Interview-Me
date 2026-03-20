import { useWebcam } from '../hooks/useWebcam';
import { InterviewState } from '../types';

interface Props {
  state: InterviewState;
}

export function UserTile({ state }: Props) {
  const { videoRef, permission } = useWebcam();
  const isListening = state === 'listening';

  return (
    <div className={`video-tile user-tile ${isListening ? 'tile--listening' : ''}`}>
      {/* 始终保留 <video> 在 DOM 中，确保 stream 到达时 ref 有效 */}
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="webcam-video"
        style={{ display: permission === 'granted' ? 'block' : 'none' }}
      />
      {permission !== 'granted' && (
        <div className="camera-placeholder">
          <span className="camera-icon">
            {permission === 'denied' ? '🚫' : '📷'}
          </span>
          <p className="camera-msg">
            {permission === 'denied'
              ? '摄像头权限被拒绝'
              : '正在获取摄像头…'}
          </p>
        </div>
      )}

      {/* 底部信息栏 */}
      <div className="tile-footer">
        <span className="tile-name">你</span>
        {isListening && (
          <span className="listening-indicator">
            <span className="listening-dot" />
            聆听中
          </span>
        )}
      </div>
    </div>
  );
}
