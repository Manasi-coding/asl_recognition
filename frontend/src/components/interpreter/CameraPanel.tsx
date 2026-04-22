import { useEffect, useRef, useState } from "react";

/**
 * Stylized hand visualization — fingertip glow dots, soft halo,
 * and a smooth trailing path following the index fingertip
 * (gradient pink → violet, fading out).
 */
const HandVisual = () => {
  const tips = [
    { x: 38, y: 38 }, // thumb
    { x: 47, y: 22 }, // index
    { x: 55, y: 17 }, // middle
    { x: 63, y: 22 }, // ring
    { x: 71, y: 30 }, // pinky
  ];
  const indexTip = tips[1];

  // Animated trail points behind index fingertip
  const [tick, setTick] = useState(0);
  const raf = useRef<number>();
  useEffect(() => {
    const start = performance.now();
    const loop = (t: number) => {
      setTick((t - start) / 1000);
      raf.current = requestAnimationFrame(loop);
    };
    raf.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf.current!);
  }, []);

  // 12 trail dots forming a smooth, slightly delayed path
  const trail = Array.from({ length: 14 }).map((_, i) => {
    const phase = tick * 1.2 - i * 0.07;
    const wave = Math.sin(phase) * 6;
    const x = indexTip.x + Math.cos(phase * 0.9) * 4;
    const y = indexTip.y - i * 1.4 + wave * 0.4 - 2;
    return { x, y, opacity: Math.max(0, 1 - i / 14) };
  });

  return (
    <div className="absolute inset-0">
      {/* Halo */}
      <div
        aria-hidden
        className="absolute left-1/2 top-1/2 h-[55%] w-[55%] -translate-x-1/2 -translate-y-[48%] rounded-full"
        style={{
          background:
            "radial-gradient(circle, hsl(339 100% 65% / 0.15), hsl(252 100% 68% / 0.08) 45%, transparent 70%)",
          filter: "blur(10px)",
        }}
      />

      {/* Faint palm */}
      <div
        aria-hidden
        className="absolute left-1/2 top-[60%] h-[28%] w-[26%] -translate-x-1/2 rounded-[40%] opacity-25"
        style={{
          background:
            "radial-gradient(ellipse, hsl(0 0% 100% / 0.18), transparent 65%)",
          filter: "blur(8px)",
        }}
      />

      {/* Connection lines (very faint) */}
      <svg className="absolute inset-0 h-full w-full opacity-25" viewBox="0 0 100 100" preserveAspectRatio="none">
        <g stroke="hsl(0 0% 100%)" strokeOpacity="0.4" strokeWidth="0.2" fill="none" strokeLinecap="round">
          {tips.map((t, i) => (
            <line key={i} x1="55" y1="60" x2={t.x} y2={t.y} />
          ))}
        </g>
      </svg>

      {/* Fingertip glow dots */}
      {tips.map((t, i) => (
        <div
          key={i}
          className="absolute -translate-x-1/2 -translate-y-1/2"
          style={{ left: `${t.x}%`, top: `${t.y}%` }}
        >
          <div
            className="h-2 w-2 rounded-full bg-white animate-dot-pulse"
            style={{
              animationDelay: `${i * 120}ms`,
              boxShadow: "0 0 10px hsl(339 100% 65% / 0.9), 0 0 22px hsl(252 100% 68% / 0.5)",
            }}
          />
        </div>
      ))}

      {/* Trail following index fingertip */}
      <svg className="absolute inset-0 h-full w-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
        <defs>
          <linearGradient id="trail-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="hsl(339 100% 65%)" stopOpacity="0.95" />
            <stop offset="1" stopColor="hsl(252 100% 68%)" stopOpacity="0" />
          </linearGradient>
        </defs>
        <polyline
          points={trail.map((p) => `${p.x},${p.y}`).join(" ")}
          stroke="url(#trail-grad)"
          strokeWidth="1.2"
          strokeLinecap="round"
          fill="none"
          style={{ filter: "drop-shadow(0 0 4px hsl(339 100% 65% / 0.6))" }}
        />
        {trail.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r={0.8 - i * 0.04} fill="hsl(339 100% 65%)" opacity={p.opacity * 0.7} />
        ))}
      </svg>
    </div>
  );
};

const PREDICT_URL = "http://localhost:8000/predict";
const CAPTURE_INTERVAL_MS = 100; // 10 FPS — matches backend movement detection threshold

export const CameraPanel = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isProcessing = useRef(false);
  const intervalRef = useRef<ReturnType<typeof setInterval>>();

  const [prediction, setPrediction] = useState<string>("");
  const [isRecording, setIsRecording] = useState(false);
  const [gestureType, setGestureType] = useState<string | null>(null);

  // ── Capture a frame and send to /predict ──────────────────────────
  const captureAndPredict = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    // Only capture when the video has enough data
    if (video.readyState !== 4) return;

    // Prevent overlapping API calls
    if (isProcessing.current) return;
    isProcessing.current = true;

    try {
      // Resize canvas to match the actual video resolution every frame
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Generate base64 JPEG at max quality — compression artifacts distort landmarks
      const base64 = canvas.toDataURL("image/jpeg", 1.0);

      const res = await fetch(PREDICT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64 }),
      });

      const json = await res.json();

      if (json.error) {
        console.warn("Predict error:", json.error);
        return;
      }

      // Update prediction + dynamic gesture state
      if (json.prediction !== undefined) {
        setPrediction(json.prediction);
      }
      setIsRecording(!!json.is_recording);
      setGestureType(json.gesture_type ?? null);
    } catch {
      // Silently skip failed frames — the next interval will retry
    } finally {
      isProcessing.current = false;
    }
  };

  // ── Camera init + prediction loop ─────────────────────────────────
  useEffect(() => {
    let cancelled = false;

    const initCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1920, height: 1080, facingMode: "user" },
          audio: false,
        });

        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }

        // Start the prediction loop once the stream is assigned
        intervalRef.current = setInterval(captureAndPredict, CAPTURE_INTERVAL_MS);
      } catch (err) {
        console.error("Camera access failed:", err);
      }
    };

    initCamera();

    return () => {
      cancelled = true;
      clearInterval(intervalRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, []);

  return (
    <div
      className="camera-inner-glow relative aspect-video w-full overflow-hidden rounded-3xl bg-black"
      style={{ borderRadius: 24 }}
    >
      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Live camera feed — full color, mirrored */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        onLoadedMetadata={(e) => {
          (e.target as HTMLVideoElement).play();
        }}
        className="absolute inset-0 h-full w-full object-cover"
        style={{ transform: "scaleX(-1)" }}
      />

      {/* Subtle dark gradient inside */}
      <div
        aria-hidden
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at center, hsl(0 0% 6%) 0%, hsl(0 0% 2%) 70%, hsl(0 0% 0%) 100%)",
          opacity: 0.3,
        }}
      />

      {/* Vignette */}
      <div
        aria-hidden
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at center, transparent 55%, hsl(0 0% 0% / 0.7) 100%)",
        }}
      />

      <HandVisual />

      {/* Top-left REC */}
      <div className="absolute left-5 top-5 flex items-center gap-2">
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-primary glow-pink-soft" />
        </span>
        <span className="font-mono text-[10.5px] uppercase tracking-[0.22em] text-foreground/80">
          Live
        </span>
      </div>

      {/* Top-right resolution */}
      <div className="absolute right-5 top-5 font-mono text-[10.5px] uppercase tracking-[0.18em] text-muted-foreground">
        1080p · 60fps
      </div>

      {/* Dynamic gesture recording indicator */}
      {isRecording && gestureType && (
        <div className="absolute left-1/2 top-14 -translate-x-1/2 flex items-center gap-2 rounded-lg border border-primary/30 bg-primary/10 px-3 py-1.5 backdrop-blur-sm">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-500/70" />
            <span className="inline-flex h-2 w-2 rounded-full bg-red-500" />
          </span>
          <span className="font-mono text-xs font-medium tracking-wider text-white">
            Recording {gestureType}…
          </span>
        </div>
      )}

      {/* Prediction HUD — bottom center */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-2 rounded-xl border border-white/10 bg-black/60 px-4 py-2 backdrop-blur-md">
        <span className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
          Sign
        </span>
        <span className="text-2xl font-bold text-white">
          {prediction || "—"}
        </span>
      </div>

      {/* Corner brackets */}
      {[
        "left-4 top-4 border-l border-t",
        "right-4 top-4 border-r border-t",
        "left-4 bottom-4 border-l border-b",
        "right-4 bottom-4 border-r border-b",
      ].map((c) => (
        <span
          key={c}
          className={`pointer-events-none absolute h-3 w-3 rounded-[2px] border-white/15 ${c}`}
        />
      ))}
    </div>
  );
};

