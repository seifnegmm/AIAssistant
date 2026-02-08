import { create } from "zustand";
import { persist } from "zustand/middleware";

interface SettingsState {
  theme: "light" | "dark" | "system";
  voiceEnabled: boolean;
  avatarEnabled: boolean;
  avatarSize: "small" | "medium" | "large";

  setTheme: (theme: "light" | "dark" | "system") => void;
  setVoiceEnabled: (enabled: boolean) => void;
  setAvatarEnabled: (enabled: boolean) => void;
  setAvatarSize: (size: "small" | "medium" | "large") => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      theme: "system",
      voiceEnabled: false,
      avatarEnabled: false,
      avatarSize: "medium",

      setTheme: (theme) => set({ theme }),
      setVoiceEnabled: (enabled) => set({ voiceEnabled: enabled }),
      setAvatarEnabled: (enabled) => set({ avatarEnabled: enabled }),
      setAvatarSize: (size) => set({ avatarSize: size }),
    }),
    {
      name: "ai-assistant-settings",
    }
  )
);
