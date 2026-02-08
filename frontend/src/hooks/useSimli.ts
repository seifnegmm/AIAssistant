"use client";

import { useRef, useCallback, useState, useEffect } from "react";
import { SimliClient } from "simli-client";

interface UseSimliConfig {
  apiKey: string;
  faceId: string;
}

interface UseSimliReturn {
  videoRef: React.RefObject<HTMLVideoElement>;
  audioRef: React.RefObject<HTMLAudioElement>;
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
  start: () => Promise<void>;
  stop: () => void;
  sendAudio: (pcm16: Uint8Array) => void;
}

export function useSimli(config: UseSimliConfig): UseSimliReturn {
  const videoRef = useRef<HTMLVideoElement>(null!);
  const audioRef = useRef<HTMLAudioElement>(null!);
  const clientRef = useRef<SimliClient | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const start = useCallback(async () => {
    if (clientRef.current || !config.apiKey || !config.faceId) return;

    setIsLoading(true);
    setError(null);

    try {
      const client = new SimliClient();

      client.Initialize({
        apiKey: config.apiKey,
        faceID: config.faceId,
        handleSilence: true,
        videoRef: videoRef.current,
        audioRef: audioRef.current,
      });

      await client.start();
      clientRef.current = client;
      setIsConnected(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Simli connection failed";
      setError(message);
      console.error("Simli start error:", err);
    } finally {
      setIsLoading(false);
    }
  }, [config.apiKey, config.faceId]);

  const stop = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.close();
      clientRef.current = null;
      setIsConnected(false);
    }
  }, []);

  const sendAudio = useCallback((pcm16: Uint8Array) => {
    if (clientRef.current && isConnected) {
      clientRef.current.sendAudioData(pcm16);
    }
  }, [isConnected]);

  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);

  return {
    videoRef,
    audioRef,
    isConnected,
    isLoading,
    error,
    start,
    stop,
    sendAudio,
  };
}
