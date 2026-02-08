import { create } from "zustand";

interface VoiceState {
  isRecording: boolean;
  isSpeaking: boolean;
  audioLevel: number;
  setRecording: (recording: boolean) => void;
  setSpeaking: (speaking: boolean) => void;
  setAudioLevel: (level: number) => void;
}

export const useVoiceStore = create<VoiceState>((set) => ({
  isRecording: false,
  isSpeaking: false,
  audioLevel: 0,
  setRecording: (recording) => set({ isRecording: recording }),
  setSpeaking: (speaking) => set({ isSpeaking: speaking }),
  setAudioLevel: (level) => set({ audioLevel: level }),
}));
