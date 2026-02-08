"use client";

import { useVoiceStore } from "@/stores/voiceStore";

interface VoiceButtonProps {
  onStart: () => void;
  onStop: () => void;
  disabled?: boolean;
}

export default function VoiceButton({
  onStart,
  onStop,
  disabled = false,
}: VoiceButtonProps) {
  const { isRecording, audioLevel } = useVoiceStore();

  const handleClick = () => {
    if (isRecording) {
      onStop();
    } else {
      onStart();
    }
  };

  const pulseScale = 1 + audioLevel * 2;

  return (
    <button
      onClick={handleClick}
      disabled={disabled}
      className={`
        relative w-16 h-16 rounded-full flex items-center justify-center
        transition-all duration-200 
        ${
          isRecording
            ? "bg-red-600 hover:bg-red-700 shadow-lg shadow-red-600/30"
            : "bg-blue-600 hover:bg-blue-700 shadow-lg shadow-blue-600/30"
        }
        ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
      `}
    >
      {/* Audio level ring */}
      {isRecording && (
        <div
          className="absolute inset-0 rounded-full border-2 border-red-400 opacity-50"
          style={{ transform: `scale(${pulseScale})`, transition: "transform 0.1s" }}
        />
      )}

      {/* Mic icon */}
      {isRecording ? (
        <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      ) : (
        <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z"
          />
        </svg>
      )}
    </button>
  );
}
