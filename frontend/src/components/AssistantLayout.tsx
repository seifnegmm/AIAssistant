"use client";

import { useState, useCallback } from "react";
import ChatPanel from "@/components/ChatPanel";
import StatusIndicator from "@/components/StatusIndicator";
import AvatarPanel from "@/components/AvatarPanel";

const SESSION_KEY = "hope_session_id";

function getOrCreateSessionId(): string {
  if (typeof window === "undefined") return crypto.randomUUID();
  const stored = localStorage.getItem(SESSION_KEY);
  if (stored) return stored;
  const id = crypto.randomUUID();
  localStorage.setItem(SESSION_KEY, id);
  return id;
}

export function AssistantLayout() {
  const [sessionId, setSessionId] = useState(getOrCreateSessionId);

  const handleNewSession = useCallback(() => {
    const id = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, id);
    setSessionId(id);
  }, []);

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-sm font-bold">
            H
          </div>
          <h1 className="text-lg font-semibold tracking-tight">Hope</h1>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={handleNewSession}
            className="px-3 py-1.5 text-xs font-medium rounded-md bg-gray-800 hover:bg-gray-700 border border-gray-700 transition-colors"
          >
            New Session
          </button>
          <StatusIndicator />
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex overflow-hidden">
        {/* Avatar Panel */}
        <div className="hidden lg:flex w-1/3 border-r border-gray-800 bg-gray-900/40">
          <AvatarPanel sessionId={sessionId} />
        </div>

        {/* Chat Panel */}
        <div className="flex-1 flex flex-col">
          <ChatPanel sessionId={sessionId} />
        </div>
      </main>
    </div>
  );
}
