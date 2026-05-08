// ── ScrubBar ──────────────────────────────────────────────────────────────────
// Shows the rolling history of detected ASL letters (real backend data only).
// Newest letter is briefly highlighted; max 15 slots visible.

import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface ScrubBarProps {
  /** Rolling list of detected letters from backend (max 15) */
  letters: string[];
}

const VISIBLE_SLOTS = 15;
const NEW_LETTER_HIGHLIGHT_MS = 1200;

export const ScrubBar = ({ letters }: ScrubBarProps) => {
  const prevLengthRef = useRef(letters.length);
  const [newestIndex, setNewestIndex] = useState<number | null>(null);

  // Detect when a new letter is appended and briefly mark it
  useEffect(() => {
    if (letters.length > prevLengthRef.current) {
      const idx = letters.length - 1;
      setNewestIndex(idx);
      const timer = setTimeout(() => setNewestIndex(null), NEW_LETTER_HIGHLIGHT_MS);
      prevLengthRef.current = letters.length;
      return () => clearTimeout(timer);
    }
    prevLengthRef.current = letters.length;
  }, [letters.length]);

  const displayLetters = letters.slice(-VISIBLE_SLOTS);
  const emptyCount = Math.max(0, VISIBLE_SLOTS - displayLetters.length);
  const offsetIdx = letters.length - displayLetters.length;

  return (
    <div className="flex h-14 items-center gap-3 rounded-2xl px-5 border border-white/[0.06] bg-black/25 backdrop-blur-md">
      {/* Label — Neon Pink */}
      <div className="flex shrink-0 items-center gap-2 pr-2">
        <span className="h-1.5 w-1.5 rounded-full bg-primary shadow-[0_0_8px_hsl(var(--primary))]" />
        <span className="font-mono text-[10px] uppercase tracking-[0.24em] text-primary">
          History
        </span>
      </div>

      <div className="mx-1 h-6 w-px bg-white/[0.08]" />

      {/* Letters — White */}
      <div className="flex flex-1 items-center justify-around gap-1 overflow-hidden">
        {displayLetters.map((letter, visIdx) => {
          const absIdx = offsetIdx + visIdx;
          const isNewest = absIdx === newestIndex;
          const opacity = 0.4 + (visIdx / (displayLetters.length - 1 || 1)) * 0.6;

          return (
            <span
              key={`${letter}-${absIdx}`}
              className={cn(
                "animate-letter-up inline-flex h-9 min-w-[34px] items-center justify-center rounded-md font-mono text-[15px] tracking-wider transition-all duration-300",
                isNewest
                  ? "scale-110 bg-primary/10 text-white ring-1 ring-primary/40 shadow-[0_0_15px_-2px_hsl(var(--primary)/0.5)]"
                  : "text-white"
              )}
              style={{
                opacity: isNewest ? 1 : opacity,
                animationDelay: `${visIdx * 15}ms`,
              }}
            >
              {letter === " " ? "·" : letter}
            </span>
          );
        })}

        {/* Empty placeholder dashes — Faint */}
        {Array.from({ length: emptyCount }).map((_, i) => (
          <span
            key={`empty-${i}`}
            className="inline-flex h-9 min-w-[34px] items-center justify-center text-[15px] text-white/10"
          >
            —
          </span>
        ))}
      </div>

      {/* Letter count badge */}
      {letters.length > 0 && (
        <div className="shrink-0 pl-2">
          <span className="font-mono text-[10px] text-white/30 tabular-nums">
            {letters.length}
          </span>
        </div>
      )}
    </div>
  );
};
