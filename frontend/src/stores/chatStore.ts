import { create } from "zustand";
import type { Message, ChatSession, ConnectionStatus } from "@/types/messages";

interface ChatState {
  messages: Message[];
  sessions: ChatSession[];
  activeSessionId: string | null;
  connectionStatus: ConnectionStatus;
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;

  addMessage: (message: Message) => void;
  updateMessage: (id: string, updates: Partial<Message>) => void;
  setMessages: (messages: Message[]) => void;
  setActiveSessionId: (id: string | null) => void;
  setConnectionStatus: (status: ConnectionStatus) => void;
  setConnected: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearMessages: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  sessions: [],
  activeSessionId: null,
  connectionStatus: "disconnected",
  isConnected: false,
  isLoading: false,
  error: null,

  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),

  updateMessage: (id, updates) =>
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, ...updates } : msg
      ),
    })),

  setMessages: (messages) => set({ messages }),
  setActiveSessionId: (id) => set({ activeSessionId: id }),
  setConnectionStatus: (status) =>
    set({ connectionStatus: status, isConnected: status === "connected" }),
  setConnected: (connected) =>
    set({
      isConnected: connected,
      connectionStatus: connected ? "connected" : "disconnected",
    }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  clearMessages: () => set({ messages: [] }),
}));
