"use client";

import { useRef, useCallback, useState } from "react";
import { useVoiceStore } from "@/stores/voiceStore";

interface UseAudioCaptureReturn {
  isRecording: boolean;
  start: () => Promise<void>;
  stop: () => void;
  error: string | null;
}

export function useAudioCapture(
  onAudioData: (pcm16: Uint8Array) => void
): UseAudioCaptureReturn {
  const { isRecording, setRecording, setAudioLevel } = useVoiceStore();
  const [error, setError] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const contextRef = useRef<AudioContext | null>(null);

  const start = useCallback(async () => {
    if (isRecording) return;
    setError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      const audioContext = new AudioContext({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);

      processor.onaudioprocess = (e) => {
        const float32 = e.inputBuffer.getChannelData(0);

        // Calculate audio level for visualization
        let sum = 0;
        for (let i = 0; i < float32.length; i++) {
          sum += float32[i] * float32[i];
        }
        setAudioLevel(Math.sqrt(sum / float32.length));

        // Convert Float32 to PCM16 (Int16)
        const pcm16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
          const s = Math.max(-1, Math.min(1, float32[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }

        onAudioData(new Uint8Array(pcm16.buffer));
      };

      source.connect(processor);
      // Connect to destination with zero-gain to keep ScriptProcessorNode alive
      // without playing captured audio back through speakers (prevents echo)
      const silentGain = audioContext.createGain();
      silentGain.gain.value = 0;
      processor.connect(silentGain);
      silentGain.connect(audioContext.destination);

      streamRef.current = stream;
      processorRef.current = processor;
      contextRef.current = audioContext;
      setRecording(true);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Microphone access denied";
      setError(message);
      console.error("Audio capture error:", err);
    }
  }, [isRecording, onAudioData, setRecording, setAudioLevel]);

  const stop = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (contextRef.current) {
      contextRef.current.close();
      contextRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setRecording(false);
    setAudioLevel(0);
  }, [setRecording, setAudioLevel]);

  return { isRecording, start, stop, error };
}
