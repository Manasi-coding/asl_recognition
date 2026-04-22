import { Pause, Square, AudioLines } from "lucide-react";
import { cn } from "@/lib/utils";

interface ControlsProps {
  isRecording: boolean;
  onToggleRecord: () => void;
  onPause?: () => void;
  elapsed: string; // "00:00:12"
}

export const Controls = ({ isRecording, onToggleRecord, onPause, elapsed }: ControlsProps) => {
  return (
    <div className="glass-strong glass-inner-highlight relative flex h-20 items-center justify-between rounded-2xl px-6">
      {/* LEFT — Listening + waveform */}
      <div className="flex items-center gap-3">
        <AudioLines className="h-4 w-4 text-muted-foreground" strokeWidth={1.75} />
        <span className="text-[12.5px] text-foreground/80">Listening...</span>
        <Waveform active={isRecording} />
      </div>

      {/* CENTER — record + pause */}
      <div className="absolute left-1/2 top-1/2 flex -translate-x-1/2 -translate-y-1/2 items-center gap-4">
        {/* Record */}
        <button
          onClick={onToggleRecord}
          aria-label={isRecording ? "Stop recording" : "Start recording"}
          className={cn(
            "press relative grid h-14 w-14 place-items-center rounded-full ring-1 transition-all",
            isRecording
              ? "ring-primary/70 animate-pulse-pink"
              : "ring-white/15 hover:ring-primary/50"
          )}
        >
          <span
            className={cn(
              "grid h-9 w-9 place-items-center rounded-full",
              isRecording ? "bg-primary text-primary-foreground" : "bg-white/[0.06] text-foreground"
            )}
          >
            {isRecording ? (
              <Square className="h-3.5 w-3.5 fill-current" strokeWidth={0} />
            ) : (
              <span className="block h-3 w-3 rounded-full bg-primary shadow-[0_0_12px_hsl(var(--primary))]" />
            )}
          </span>
        </button>

        {/* Pause */}
        <button
          onClick={onPause}
          aria-label="Pause"
          className="press grid h-12 w-12 place-items-center rounded-full bg-white/[0.04] text-foreground/85 ring-1 ring-white/10 hover:bg-white/[0.08] hover:text-foreground"
        >
          <Pause className="h-4 w-4 fill-current" strokeWidth={0} />
        </button>
      </div>

      {/* RIGHT — timer badge */}
      <div className="flex items-center gap-2.5">
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-primary shadow-[0_0_8px_hsl(var(--primary))]" />
        </span>
        <div className="flex flex-col leading-tight">
          <span className="font-mono text-[13px] text-foreground tracking-wider">{elapsed}</span>
          <span className="text-[10px] text-primary">Recording</span>
        </div>
      </div>
    </div>
  );
};

const Waveform = ({ active }: { active: boolean }) => {
  const bars = [0.3, 0.55, 0.4, 0.75, 0.5, 0.85, 0.45, 0.7, 0.5, 0.6, 0.4, 0.55, 0.35, 0.65, 0.45, 0.75, 0.5, 0.4, 0.55, 0.35];
  return (
    <div className="flex h-6 w-44 items-center gap-[2.5px]">
      {bars.map((h, i) => (
        <span
          key={i}
          className={cn(
            "block w-[2px] rounded-full bg-primary/70 transition-opacity",
            active ? "animate-wave opacity-90" : "opacity-30"
          )}
          style={{
            height: `${h * 100}%`,
            animationDelay: `${i * 60}ms`,
            animationDuration: `${800 + (i % 5) * 110}ms`,
          }}
        />
      ))}
    </div>
  );
};
