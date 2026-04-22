import { cn } from "@/lib/utils";

interface ScrubBarProps {
  letters: string[];
  activeIndex: number;
  /** Total slots to display (empty placeholders shown as dashes) */
  slots?: number;
}

export const ScrubBar = ({ letters, activeIndex, slots = 8 }: ScrubBarProps) => {
  const filled = letters.slice(0, slots);
  const emptyCount = Math.max(0, slots - filled.length);

  return (
    <div className="glass-strong glass-inner-highlight flex h-14 items-center gap-3 rounded-2xl px-5">
      {/* Left label */}
      <div className="flex items-center gap-2 pr-2">
        <span className="h-1.5 w-1.5 rounded-full bg-primary shadow-[0_0_8px_hsl(var(--primary))]" />
        <span className="font-mono text-[10px] uppercase tracking-[0.24em] text-muted-foreground">
          Scrub
        </span>
      </div>

      <div className="mx-1 h-6 w-px bg-white/[0.08]" />

      {/* Letters */}
      <div className="flex flex-1 items-center justify-around gap-1">
        {filled.map((l, i) => {
          const isActive = i === activeIndex;
          const isPast = i < activeIndex;
          return (
            <span
              key={`${l}-${i}`}
              className={cn(
                "animate-letter-up inline-flex h-9 min-w-[34px] items-center justify-center rounded-md font-mono text-[15px] tracking-wider transition-all duration-300",
                isActive
                  ? "scale-110 bg-primary/15 text-foreground ring-1 ring-primary/60 shadow-[0_0_18px_-2px_hsl(var(--primary)/0.7)]"
                  : isPast
                    ? "text-foreground/85"
                    : "text-muted-foreground/55"
              )}
              style={{ animationDelay: `${i * 25}ms` }}
            >
              {l === " " ? "·" : l}
            </span>
          );
        })}

        {/* Empty placeholder dashes */}
        {Array.from({ length: emptyCount }).map((_, i) => (
          <span
            key={`empty-${i}`}
            className="inline-flex h-9 min-w-[34px] items-center justify-center text-[15px] text-muted-foreground/25"
          >
            —
          </span>
        ))}
      </div>
    </div>
  );
};
