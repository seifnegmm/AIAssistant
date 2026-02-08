export interface AvatarState {
  isActive: boolean;
  isLoading: boolean;
  isSpeaking: boolean;
  error: string | null;
}

export interface SimliConfig {
  apiKey: string;
  faceId: string;
  handleSilence: boolean;
  maxSessionLength: number;
  maxIdleTime: number;
}
