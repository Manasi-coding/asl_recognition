// ── usePredictionLoop ─────────────────────────────────────────────────────────
// Owns the webcam stream, the capture interval, and all prediction state.
// Exposes start() / stop() for explicit session control.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEFAULT_CONFIG,
  fetchConfig,
  postPredict,
  type AppConfig,
  type PredictResponse,
} from "@/api";

const VALID_GESTURE_TYPES = new Set<string>(["J", "Z"]);
const MAX_HISTORY = 15;

export type ConnectionStatus = "connected" | "reconnecting" | "offline";

export interface PredictionState {
  /** Current raw prediction from backend (empty string = no hand) */
  currentPrediction: string;
  /** Rolling history of confirmed, changed letters (max 15) */
  recentPredictions: string[];
  /** Backend is actively recording a dynamic gesture */
  isRecording: boolean;
  /** Which gesture is being recorded ("J" | "Z" | null) */
  gestureType: "J" | "Z" | null;
  /** Backend reachability */
  backendConnected: ConnectionStatus;
  /** Config loaded from backend */
  config: AppConfig;
  /** Whether the session is active (camera + polling running) */
  isActive: boolean;
}

export interface PredictionLoopControls {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  streamReady: boolean;
  clearHistory: () => void;
  /** Start webcam + prediction loop */
  start: () => void;
  /** Stop webcam + prediction loop */
  stop: () => void;
}

export function usePredictionLoop(): PredictionState & PredictionLoopControls {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isProcessing = useRef(false);
  const intervalRef = useRef<ReturnType<typeof setInterval>>();
  const lastPredictionRef = useRef<string>("");

  const [config, setConfig] = useState<AppConfig>(DEFAULT_CONFIG);
  const [streamReady, setStreamReady] = useState(false);
  const [isActive, setIsActive] = useState(false);
  const [currentPrediction, setCurrentPrediction] = useState("");
  const [recentPredictions, setRecentPredictions] = useState<string[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [gestureType, setGestureType] = useState<"J" | "Z" | null>(null);
  const [backendConnected, setBackendConnected] =
    useState<ConnectionStatus>("reconnecting");

  // ── Load backend config on mount (once) ───────────────────────────────────
  useEffect(() => {
    fetchConfig()
      .then((cfg) => {
        setConfig(cfg);
        setBackendConnected("connected");
      })
      .catch(() => {
        console.warn("Could not load backend config — using defaults");
        setBackendConnected("offline");
      });
  }, []);

  // ── Frame capture + predict ───────────────────────────────────────────────
  const captureAndPredict = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState !== 4) return;
    if (isProcessing.current) return;
    isProcessing.current = true;

    try {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // Mirror horizontally — backend receives the same orientation the user sees
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.setTransform(1, 0, 0, 1, 0, 0);

      const base64 = canvas.toDataURL("image/jpeg", 1.0);
      const data: PredictResponse = await postPredict(base64);

      setBackendConnected("connected");

      const { prediction, is_recording, gesture_type } = data;

      setCurrentPrediction(prediction);
      setIsRecording(!!is_recording);

      const gt =
        gesture_type && VALID_GESTURE_TYPES.has(gesture_type)
          ? (gesture_type as "J" | "Z")
          : null;
      setGestureType(gt);

      // Append to history only when prediction changes and is non-empty
      if (prediction && prediction !== lastPredictionRef.current) {
        lastPredictionRef.current = prediction;
        setRecentPredictions((prev) => {
          const next = [...prev, prediction];
          return next.length > MAX_HISTORY
            ? next.slice(next.length - MAX_HISTORY)
            : next;
        });
      }
    } catch {
      setBackendConnected((prev) =>
        prev === "connected" ? "reconnecting" : "offline"
      );
    } finally {
      isProcessing.current = false;
    }
  }, []);

  // ── Start session ─────────────────────────────────────────────────────────
  const start = useCallback(async () => {
    if (streamRef.current) return; // already running

    const constraints = [
      { video: { width: 1920, height: 1080, facingMode: "user" }, audio: false },
      { video: { width: 1280, height: 720, facingMode: "user" }, audio: false },
      { video: true, audio: false }
    ];

    let lastErr: any = null;

    for (const constraint of constraints) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraint);
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
        setStreamReady(true);
        setIsActive(true);
        return; // Success!
      } catch (err) {
        lastErr = err;
        console.warn(`Camera attempt failed with constraints:`, constraint, err);
        // Continue to next fallback
      }
    }

    // If we get here, all fallbacks failed
    console.error("All camera access attempts failed:", lastErr);
    if (lastErr?.name === "NotReadableError") {
      alert("Camera Error: The camera is already in use by another application or tab. Please close other apps using the camera and try again.");
    } else {
      alert(`Camera Access Failed: ${lastErr?.message || "Unknown error"}`);
    }
    setIsActive(false);
  }, []);

  // ── Stop session ──────────────────────────────────────────────────────────
  const stop = useCallback(() => {
    clearInterval(intervalRef.current);
    intervalRef.current = undefined;

    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;

    if (videoRef.current) videoRef.current.srcObject = null;

    setStreamReady(false);
    setIsActive(false);
    setCurrentPrediction("");
    setIsRecording(false);
    setGestureType(null);
    lastPredictionRef.current = "";
    isProcessing.current = false;
  }, []);

  // ── Start interval when stream becomes ready ──────────────────────────────
  useEffect(() => {
    if (!streamReady) return;
    intervalRef.current = setInterval(
      captureAndPredict,
      config.capture_interval_ms
    );
    return () => clearInterval(intervalRef.current);
  }, [streamReady, config.capture_interval_ms, captureAndPredict]);

  // ── Cleanup on unmount ────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      clearInterval(intervalRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const clearHistory = useCallback(() => {
    setRecentPredictions([]);
    lastPredictionRef.current = "";
  }, []);

  return {
    videoRef,
    canvasRef,
    streamReady,
    isActive,
    currentPrediction,
    recentPredictions,
    isRecording,
    gestureType,
    backendConnected,
    config,
    clearHistory,
    start,
    stop,
  };
}
