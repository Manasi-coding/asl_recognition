// ── CameraPanel ───────────────────────────────────────────────────────────────
// Renders the webcam video + overlaid history bar.
// Everything here is positioned absolutely inside the parent's relative context.

import { type PredictionLoopControls, type PredictionState } from "@/hooks/usePredictionLoop";
import { ScrubBar } from "./ScrubBar";
import { cn } from "@/lib/utils";

type CameraPanelProps = Pick<
  PredictionState & PredictionLoopControls,
  | "videoRef"
  | "canvasRef"
  | "currentPrediction"
  | "isRecording"
  | "gestureType"
  | "config"
  | "streamReady"
  | "isActive"
> & {
  recentPredictions: string[];
};

export const CameraPanel = ({
  videoRef,
  canvasRef,
  currentPrediction,
  isRecording,
  gestureType,
  config,
  streamReady,
  isActive,
  recentPredictions,
}: CameraPanelProps) => {
  return (
    <>
      {/* Hidden canvas — frame capture only */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Live webcam feed — fills the parent container */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        onLoadedMetadata={(e) => (e.target as HTMLVideoElement).play()}
        className="absolute inset-0 h-full w-full object-cover"
        style={{ transform: "scaleX(-1)" }}
      />

      {/* Cinematic grain / noise overlay */}
      <div className="pointer-events-none absolute inset-0 opacity-[0.03] bg-[url('https://grainy-gradients.vercel.app/noise.svg')]" />

      {/* Cinematic vignette — darkens edges */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at center, transparent 45%, hsl(0 0% 0% / 0.6) 100%)",
        }}
      />

      {/* Camera not ready / session inactive overlay */}
      {!streamReady && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-black/60 backdrop-blur-[2px]">
          {isActive ? (
            <>
              <div className="h-8 w-8 rounded-full border-2 border-primary border-t-transparent animate-spin" />
              <span className="font-mono text-xs tracking-widest text-muted-foreground uppercase">
                Initialising camera…
              </span>
            </>
          ) : (
            <span className="font-mono text-[11px] tracking-[0.2em] text-muted-foreground/40 uppercase">
              Standby
            </span>
          )}
        </div>
      )}

      {/* ── HUD overlays ── */}

      {/* Dynamic gesture recording banner */}
      {isRecording && gestureType && (
        <div className="absolute left-1/2 top-16 -translate-x-1/2 flex items-center gap-2 rounded-lg border border-primary/30 bg-primary/10 px-3 py-1.5 backdrop-blur-sm animate-fade-in">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-500/70" />
            <span className="inline-flex h-2 w-2 rounded-full bg-red-500" />
          </span>
          <span className="font-mono text-xs font-medium tracking-wider text-white">
            Recording {gestureType} gesture…
          </span>
        </div>
      )}

      {/* ── OVERLAID HISTORY BAR ── */}
      <div className="absolute bottom-5 left-5 right-5 z-20">
        <ScrubBar letters={recentPredictions} />
      </div>

      {/* Corner bracket accents */}
      {[
        "left-4 top-4 border-l border-t",
        "right-4 top-4 border-r border-t",
        "left-4 bottom-4 border-l border-b",
        "right-4 bottom-4 border-r border-b",
      ].map((c) => (
        <span
          key={c}
          className={`pointer-events-none absolute h-3 w-3 rounded-[2px] border-white/[0.08] ${c}`}
        />
      ))}
    </>
  );
};
