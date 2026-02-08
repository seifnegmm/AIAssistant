"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Message, ServerMessage } from "@/types/messages";
import { useChatStore } from "@/stores/chatStore";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
const RECONNECT_DELAY_MS = 3000;
const MAX_RECONNECT_ATTEMPTS = 5;

type BinaryHandler = (data: Uint8Array) => void;

export function useWebSocket(sessionId: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimer = useRef<NodeJS.Timeout | null>(null);
  const binaryHandlerRef = useRef<BinaryHandler | null>(null);
  const [status, setStatus] = useState<
    "connecting" | "connected" | "disconnected" | "error"
  >("disconnected");

  const {
    addMessage,
    updateMessage,
    setConnectionStatus,
    setConnected,
    setLoading,
    setError,
  } = useChatStore();

  /** Register a callback for incoming binary (TTS audio) messages. */
  const registerOnBinary = useCallback((handler: BinaryHandler | null) => {
    binaryHandlerRef.current = handler;
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus("connecting");
    setConnectionStatus("connecting");
    const ws = new WebSocket(`${WS_URL}/ws/chat?session_id=${sessionId}`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      setStatus("connected");
      setConnectionStatus("connected");
      setConnected(true);
      setError(null);
      reconnectAttempts.current = 0;
    };

    ws.onmessage = (event) => {
      // Handle binary audio data (TTS PCM16) from backend
      if (event.data instanceof ArrayBuffer) {
        if (binaryHandlerRef.current) {
          binaryHandlerRef.current(new Uint8Array(event.data));
        }
        return;
      }

      // Handle JSON text messages
      if (typeof event.data === "string") {
        try {
          const data: ServerMessage = JSON.parse(event.data);

          switch (data.type) {
            case "text": {
              const assistantMsg: Message = {
                id: crypto.randomUUID(),
                role: "assistant",
                content: data.content || "",
                timestamp: new Date().toISOString(),
              };
              addMessage(assistantMsg);
              setLoading(false);
              break;
            }
            case "audio": {
              // Base64-encoded audio from backend — decode and forward to binary handler
              if (data.content && binaryHandlerRef.current) {
                const raw = atob(data.content);
                const bytes = new Uint8Array(raw.length);
                for (let i = 0; i < raw.length; i++) {
                  bytes[i] = raw.charCodeAt(i);
                }
                binaryHandlerRef.current(bytes);
              }
              break;
            }
            case "transcript":
              // STT transcript — could update the last user message
              break;
            case "status":
              if (data.content === "thinking" || data.content === "Thinking...") {
                setLoading(true);
              }
              break;
            case "error":
              setError(data.content || "Unknown error");
              setLoading(false);
              break;
          }
        } catch (err) {
          console.error("Failed to parse WebSocket message:", err);
        }
      }
    };

    ws.onclose = () => {
      setStatus("disconnected");
      setConnectionStatus("disconnected");
      setConnected(false);

      if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
        reconnectTimer.current = setTimeout(() => {
          reconnectAttempts.current += 1;
          connect();
        }, RECONNECT_DELAY_MS);
      }
    };

    ws.onerror = () => {
      setStatus("error");
      setConnectionStatus("error");
      setError("WebSocket connection error");
    };

    wsRef.current = ws;
  }, [sessionId, addMessage, setConnected, setConnectionStatus, setLoading, setError]);

  const disconnect = useCallback(() => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
    }
    reconnectAttempts.current = MAX_RECONNECT_ATTEMPTS; // prevent reconnect
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  const sendTextMessage = useCallback(
    (content: string) => {
      if (wsRef.current?.readyState !== WebSocket.OPEN) return;

      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content,
        timestamp: new Date().toISOString(),
      };
      addMessage(userMsg);
      setLoading(true);

      wsRef.current.send(
        JSON.stringify({
          type: "text",
          content,
        })
      );
    },
    [addMessage, setLoading]
  );

  const sendAudioData = useCallback((audioData: ArrayBuffer) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(new Uint8Array(audioData));
  }, []);

  /** Signal the backend that the user stopped recording so it can STT the buffered audio. */
  const sendStopAudio = useCallback(() => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ type: "stop_audio" }));
  }, []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return {
    status,
    sendTextMessage,
    sendAudioData,
    sendStopAudio,
    registerOnBinary,
    disconnect,
    reconnect: connect,
  };
}
