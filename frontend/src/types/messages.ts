export type MessageRole = "user" | "assistant" | "system";

export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: string;
  isStreaming?: boolean;
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
}

export interface ChatSession {
  session_id: string;
  created_at: string;
  last_active: string;
  message_count: number;
  summary: string | null;
}

export interface ServerMessage {
  type: "text" | "stream" | "transcript" | "status" | "error";
  content?: string;
  session_id?: string;
  timestamp?: string;
  is_final?: boolean;
}

export interface WebSocketTextMessage {
  type: "text";
  content: string;
}

export interface WebSocketAudioMessage {
  type: "audio";
  audio_data: string;
}

export type WebSocketOutgoingMessage = WebSocketTextMessage | WebSocketAudioMessage;

export type WebSocketIncomingMessage = ServerMessage;
