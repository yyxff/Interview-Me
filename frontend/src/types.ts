export type MessageRole = 'user' | 'assistant';

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
}

export type InterviewState =
  | 'idle'        // 未开始
  | 'listening'   // 监听中（等待用户说话）
  | 'processing'  // 请求后端中
  | 'speaking';   // AI 正在朗读
