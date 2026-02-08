const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface HealthResponse {
  status: string;
  version: string;
  services: Record<string, string>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health`, {
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchSessions(): Promise<unknown[]> {
  const res = await fetch(`${API_BASE}/api/sessions`);
  if (!res.ok) {
    throw new Error(`Failed to fetch sessions: ${res.status}`);
  }
  return res.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/sessions/${sessionId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(`Failed to delete session: ${res.status}`);
  }
}

export { API_BASE };
