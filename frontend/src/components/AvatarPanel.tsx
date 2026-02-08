"use client";

import { useState, useCallback, useEffect } from "react";
import SimliAvatar from "./SimliAvatar";
import VoiceButton from "./VoiceButton";
import { useAudioCapture } from "@/hooks/useAudioCapture";
import { useWebSocket } from "@/hooks/useWebSocket";

interface AvatarPanelProps {
  sessionId: string;
}

export default function AvatarPanel({ sessionId }: AvatarPanelProps) {
  const [avatarActive, setAvatarActive] = useState(false);
  const [audioData, setAudioData] = useState<Uint8Array | null>(null);

  const simliApiKey = process.env.NEXT_PUBLIC_SIMLI_API_KEY || "";
  const simliFaceId = process.env.NEXT_PUBLIC_SIMLI_FACE_ID || "";

  const { sendAudioData, sendStopAudio, status, registerOnBinary } = useWebSocket(sessionId);

  // Wire incoming TTS audio from WebSocket to SimliAvatar
  useEffect(() => {
    registerOnBinary((pcm16: Uint8Array) => {
      setAudioData(pcm16);
    });
    return () => registerOnBinary(null);
  }, [registerOnBinary]);

  const handleMicAudio = useCallback(
    (pcm16: Uint8Array) => {
      sendAudioData(pcm16);
    },
    [sendAudioData]
  );

  const { isRecording, start: startRecording, stop: stopRecording, error: micError } =
    useAudioCapture(handleMicAudio);

  const handleStopRecording = useCallback(() => {
    stopRecording();
    sendStopAudio();
  }, [stopRecording, sendStopAudio]);

  const hasSimli = !!simliApiKey && !!simliFaceId;

  return (
    <div className="flex flex-col h-full">
      {/* Avatar viewport */}
      <div className="flex-1 relative">
        {hasSimli ? (
          <SimliAvatar
            apiKey={simliApiKey}
            faceId={simliFaceId}
            isActive={avatarActive}
            audioData={audioData}
          />
        ) : (
          <div className="w-full h-full bg-gray-900 rounded-2xl flex items-center justify-center">
            <div className="text-center px-6">
              <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-gray-800 flex items-center justify-center">
                <svg
                  className="w-12 h-12 text-gray-500"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0"
                  />
                </svg>
              </div>
              <p className="text-gray-400 text-sm">
                Set <code className="text-blue-400">SIMLI_API_KEY</code> and{" "}
                <code className="text-blue-400">SIMLI_FACE_ID</code> to enable avatar
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Controls bar */}
      <div className="flex items-center justify-center gap-4 py-4">
        {hasSimli && (
          <button
            onClick={() => setAvatarActive(!avatarActive)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              avatarActive
                ? "bg-green-600 hover:bg-green-700 text-white"
                : "bg-gray-700 hover:bg-gray-600 text-gray-300"
            }`}
          >
            {avatarActive ? "Avatar On" : "Avatar Off"}
          </button>
        )}

        <VoiceButton
          onStart={startRecording}
          onStop={handleStopRecording}
          disabled={status !== "connected"}
        />

        {micError && (
          <p className="text-red-400 text-xs max-w-[200px]">{micError}</p>
        )}
      </div>
    </div>
  );
}
