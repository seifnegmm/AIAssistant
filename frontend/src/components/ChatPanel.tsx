"use client";

import { useRef, useEffect, useState, useCallback, FormEvent, KeyboardEvent } from "react";
import { useChatStore } from "@/stores/chatStore";
import MessageBubble from "./MessageBubble";
import { useWebSocket } from "@/hooks/useWebSocket";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ChatPanelProps {
  sessionId: string;
}

export default function ChatPanel({ sessionId }: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const { messages, isLoading, setMessages, clearMessages } = useChatStore();

  const { status, sendTextMessage } = useWebSocket(sessionId);

  // Load chat history from backend on mount / session change
  const loadHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/sessions/${sessionId}/history`);
      if (!res.ok) return;
      const data = await res.json();
      if (data.messages && data.messages.length > 0) {
        const mapped = data.messages.map((m: { role: string; content: string; timestamp: string }, i: number) => ({
          id: `history-${i}`,
          role: m.role === "human" ? "user" : m.role === "ai" ? "assistant" : m.role,
          content: m.content,
          timestamp: m.timestamp,
        }));
        setMessages(mapped);
      }
    } catch {
      /* no-op: start fresh */
    } finally {
      setHistoryLoaded(true);
    }
  }, [sessionId, setMessages]);

  useEffect(() => {
    clearMessages();
    setHistoryLoaded(false);
    loadHistory();
  }, [sessionId, loadHistory, clearMessages]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading || status !== "connected") return;

    // useWebSocket.sendTextMessage already adds the user message to the store
    sendTextMessage(trimmed);
    setInput("");
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-950">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500">
              <p className="text-4xl mb-4">👋</p>
              <p className="text-lg font-medium">Hello! How can I help you?</p>
              <p className="text-sm mt-1">Type a message to start chatting.</p>
            </div>
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {isLoading && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-800 rounded-2xl rounded-bl-md px-4 py-3">
              <div className="flex gap-1">
                <span className="h-2 w-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
                <span className="h-2 w-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
                <span className="h-2 w-2 bg-gray-400 rounded-full animate-bounce" />
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        className="px-6 py-4 border-t border-gray-800"
      >
        <div className="flex gap-3 items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message..."
            rows={1}
            className="flex-1 resize-none rounded-xl bg-gray-800 px-4 py-3 text-sm text-white placeholder-gray-500 outline-none focus:ring-2 focus:ring-blue-600 max-h-32"
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading || status !== "connected"}
            className="rounded-xl bg-blue-600 px-5 py-3 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
