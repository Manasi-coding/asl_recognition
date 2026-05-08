import { Play } from "lucide-react";
import { cn } from "@/lib/utils";
import { NeonHand } from "./NeonHand";

interface LetterCardProps {
  letter: string;
}

export const LetterCard = ({ letter }: LetterCardProps) => {
  return (
    <div
      className={cn(
        "press group relative overflow-hidden rounded-2xl bg-[#0a0a0a] border border-white/[0.06] p-4",
        "transition-all duration-300 hover:border-primary/25 hover:shadow-[0_0_22px_rgba(255,46,140,0.18)]"
      )}
    >
      {/* Subtle corner bloom on hover */}
      <div className="pointer-events-none absolute -right-6 -top-6 h-28 w-28 rounded-full bg-primary/5 blur-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

      {/* Top row: letter label + play button */}
      <div className="relative z-10 flex items-center justify-between mb-1">
        <span className="font-mono text-[13px] font-semibold tracking-widest text-white/80">
          {letter}
        </span>
        <button
          aria-label={`Play ${letter} sign`}
          className="press flex h-7 w-7 items-center justify-center rounded-full bg-white/[0.03] text-white/30 border border-white/[0.07] transition-all duration-200 group-hover:text-primary group-hover:border-primary/30 group-hover:bg-primary/[0.06]"
        >
          <Play className="h-2.5 w-2.5 fill-current" strokeWidth={0} />
        </button>
      </div>

      {/* Neon Hand Sign Illustration */}
      <div className="relative aspect-square w-full">
        <NeonHand letter={letter} />
      </div>
    </div>
  );
};
