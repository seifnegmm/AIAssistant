export interface ChatSession {
  session_id: string;
  created_at: string;
  last_active: string;
  message_count: number;
  summary: string | null;
}

export interface ChatState {
  sessionId: string;
  messages: import("./messages").ChatMessage[];
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
}

export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";
