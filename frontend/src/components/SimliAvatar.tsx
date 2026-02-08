"use client";

import { useEffect, useRef } from "react";
import { useSimli } from "@/hooks/useSimli";

interface SimliAvatarProps {
  apiKey: string;
  faceId: string;
  isActive: boolean;
  audioData: Uint8Array | null;
}

export default function SimliAvatar({
  apiKey,
  faceId,
  isActive,
  audioData,
}: SimliAvatarProps) {
  const { videoRef, audioRef, isConnected, isLoading, error, start, stop, sendAudio } =
    useSimli({ apiKey, faceId });

  // Track previous isActive to detect transitions
  const prevActiveRef = useRef(isActive);

  useEffect(() => {
    if (isActive && !isConnected && !isLoading) {
      start();
    } else if (!isActive && isConnected) {
      stop();
    }
    prevActiveRef.current = isActive;
  }, [isActive, isConnected, isLoading, start, stop]);

  useEffect(() => {
    if (audioData && isConnected) {
      sendAudio(audioData);
    }
  }, [audioData, isConnected, sendAudio]);

  return (
    <div className="relative w-full h-full bg-gray-900 rounded-2xl overflow-hidden">
      {/* Video element for avatar */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted={false}
        className="w-full h-full object-cover"
      />

      {/* Hidden audio element for Simli SDK */}
      <audio ref={audioRef} autoPlay />

      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
          <div className="flex flex-col items-center gap-3">
            <div className="w-10 h-10 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-gray-300">Connecting avatar...</span>
          </div>
        </div>
      )}

      {/* Error overlay */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
          <div className="text-center px-4">
            <p className="text-red-400 text-sm">{error}</p>
            <button
              onClick={start}
              className="mt-2 px-4 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* Idle state (not connected, not loading) */}
      {!isConnected && !isLoading && !error && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
          <div className="text-center">
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
            <p className="text-gray-400 text-sm">Avatar inactive</p>
          </div>
        </div>
      )}
    </div>
  );
}
