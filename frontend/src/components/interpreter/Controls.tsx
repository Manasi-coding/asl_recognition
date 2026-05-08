// ── Controls ──────────────────────────────────────────────────────────────────
// Bottom status bar with session controls and stats.
// LEFT: Letters count | CENTER: Session controls | RIGHT: Session timer + Clear

import { Trash2, Play, Square } from "lucide-react";
import { cn } from "@/lib/utils";

interface ControlsProps {
  currentPrediction: string;
  letterCount: number;
  onClear: () => void;
  onStart: () => void;
  onStop: () => void;
  isRecording: boolean;
  isActive: boolean;
  elapsed: string;
}

export const Controls = ({
  currentPrediction,
  letterCount,
  onClear,
  onStart,
  onStop,
  isRecording,
  isActive,
  elapsed,
}: ControlsProps) => {
  return (
    <div className="glass-strong glass-inner-highlight relative flex h-24 shrink-0 items-center justify-between rounded-3xl px-8 border border-white/10">

      {/* LEFT — Letters count (Neon Pink Label, White Number) */}
      <div className="flex flex-col items-start gap-1 min-w-[80px]">
        <span className="font-mono text-[11px] uppercase tracking-[0.2em] text-primary">
          Letters
        </span>
        <span className="font-mono text-[26px] font-bold leading-none text-white tabular-nums">
          {letterCount}
        </span>
      </div>

      {/* CENTER — Session Controls + Prediction Indicator */}
      <div className="absolute left-1/2 top-1/2 flex -translate-x-1/2 -translate-y-1/2 flex-col items-center gap-3">
        {/* Prediction Indicator — Minimal & Integrated */}
        <div className="flex items-center gap-3 h-6">
          <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground/40">
            Current:
          </span>
          <span className="font-mono text-[18px] font-bold text-white transition-all duration-200">
            {isActive && currentPrediction ? currentPrediction : "—"}
          </span>
        </div>

        <div className="flex items-center gap-3">
          {/* START Button — Outlined Neon Pink Pill */}
          <button
            onClick={onStart}
            disabled={isActive}
            className={cn(
              "press flex items-center gap-2 rounded-full border border-primary/60 px-6 py-2 transition-all duration-300",
              "text-[12px] font-mono uppercase tracking-[0.15em]",
              isActive
                ? "opacity-20 cursor-not-allowed border-white/10 text-muted-foreground"
                : "text-primary shadow-[0_0_15px_hsl(var(--primary)/0.3)] hover:bg-primary/5 hover:shadow-[0_0_25px_hsl(var(--primary)/0.5)]"
            )}
          >
            <Play className="h-3 w-3 fill-current" />
            Start
          </button>

          {/* STOP Button — Darker Muted Pill */}
          <button
            onClick={onStop}
            disabled={!isActive}
            className={cn(
              "press flex items-center gap-2 rounded-full border border-white/10 px-6 py-2 transition-all duration-300",
              "text-[12px] font-mono uppercase tracking-[0.15em]",
              !isActive
                ? "opacity-20 cursor-not-allowed text-muted-foreground"
                : "bg-white/[0.04] text-foreground/80 hover:bg-white/[0.08] hover:text-white"
            )}
          >
            <Square className="h-3 w-3 fill-primary shadow-[0_0_8px_hsl(var(--primary))]" strokeWidth={0} />
            Stop
          </button>
        </div>
      </div>

      {/* RIGHT — Session Timer + Clear (Neon Pink Label, White Numbers) */}
      <div className="flex items-center gap-6">
        <div className="flex flex-col items-end gap-1">
          <span className="font-mono text-[11px] uppercase tracking-[0.2em] text-primary">
            Session
          </span>
          <span className="font-mono text-[18px] font-medium tracking-widest text-white tabular-nums">
            {elapsed}
          </span>
        </div>

        {/* Clear Button */}
        <button
          onClick={onClear}
          aria-label="Clear letter history"
          disabled={letterCount === 0}
          className="press grid h-10 w-10 place-items-center rounded-full bg-white/[0.03] text-muted-foreground/60 border border-white/5 hover:bg-white/[0.08] hover:text-white disabled:opacity-20 transition-all"
        >
          <Trash2 className="h-4 w-4" strokeWidth={1.5} />
        </button>
      </div>
    </div>
  );
};
