"use client";

import { useChatStore } from "@/stores/chatStore";
import type { ConnectionStatus } from "@/types/messages";

const STATUS_CONFIG: Record<
  ConnectionStatus,
  { color: string; label: string }
> = {
  connected: { color: "bg-green-500", label: "Connected" },
  connecting: { color: "bg-yellow-500", label: "Connecting..." },
  disconnected: { color: "bg-red-500", label: "Disconnected" },
  error: { color: "bg-red-600", label: "Error" },
};

export default function StatusIndicator() {
  const connectionStatus = useChatStore((s) => s.connectionStatus);
  const config = STATUS_CONFIG[connectionStatus] ?? STATUS_CONFIG.disconnected;

  return (
    <div className="flex items-center gap-2">
      <span
        className={`h-2 w-2 rounded-full ${config.color} ${
          connectionStatus === "connecting" ? "animate-pulse" : ""
        }`}
      />
      <span className="text-xs text-gray-400">{config.label}</span>
    </div>
  );
}
