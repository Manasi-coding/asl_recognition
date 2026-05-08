// ── Signify — typed API client ───────────────────────────────────────────────
// All communication with http://127.0.0.1:8000 goes through this module.

export const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

// ── Response types ────────────────────────────────────────────────────────────

export interface PredictResponse {
  prediction: string;
  is_recording: boolean;
  gesture_type: "J" | "Z" | null;
  cooldown: number;
}

export interface AppConfig {
  capture_interval_ms: number;
  movement_threshold: number;
  still_frames_threshold: number;
  cooldown_frames: number;
  min_sequence_length: number;
  labels: {
    no_hand: string;
    scanning: string;
  };
}

export interface ClassesResponse {
  classes: string[];
}

// ── Default config (used before backend responds) ─────────────────────────────

export const DEFAULT_CONFIG: AppConfig = {
  capture_interval_ms: 100,
  movement_threshold: 0.02,
  still_frames_threshold: 5,
  cooldown_frames: 5,
  min_sequence_length: 20,
  labels: { no_hand: "", scanning: "—" },
};

// ── API calls ─────────────────────────────────────────────────────────────────

export async function fetchConfig(): Promise<AppConfig> {
  const res = await fetch(`${API_URL}/api/config`);
  if (!res.ok) throw new Error(`Config fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchClasses(): Promise<string[]> {
  const res = await fetch(`${API_URL}/api/classes`);
  if (!res.ok) throw new Error(`Classes fetch failed: ${res.status}`);
  const data: ClassesResponse = await res.json();
  return data.classes ?? [];
}

export async function postPredict(base64Image: string): Promise<PredictResponse> {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: base64Image }),
  });
  if (!res.ok) throw new Error(`Predict failed: ${res.status}`);
  const json = await res.json();
  if (json.error) throw new Error(`Backend error: ${json.error}`);
  return json as PredictResponse;
}
