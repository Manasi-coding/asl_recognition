import { MoreHorizontal } from "lucide-react";
import { cn } from "@/lib/utils";
import type { HistoryItem } from "@/hooks/useHistoryStore";

// Letters whose illustrations still need extra breathing room
const SCALE_OVERRIDES: Record<string, number> = {
  B: 0.88, F: 0.88, K: 0.88, L: 0.88, U: 0.88, V: 0.88, W: 0.88,
};

interface HistoryLetterCardProps {
  item: HistoryItem;
}

export const HistoryLetterCard = ({ item }: HistoryLetterCardProps) => {
  const { letter, timestamp, accuracy } = item;
  const scale = SCALE_OVERRIDES[letter] ?? 1;

  return (
    <div
      className={cn(
        "group relative flex flex-col overflow-hidden rounded-xl",
        "bg-black border border-white/[0.07]",
        "transition-all duration-300",
        "hover:border-primary/25 hover:shadow-[0_0_16px_rgba(255,45,140,0.12)]"
      )}
    >
      {/* Top row: letter label + 3-dot menu */}
      <div className="flex items-start justify-between px-3 pt-3 pb-0">
        <span className="font-mono text-[11px] font-semibold tracking-[0.2em] text-white/60 uppercase">
          {letter}
        </span>
        <button
          aria-label="More options"
          className="grid h-5 w-5 place-items-center rounded-full text-muted-foreground/40
                     transition-colors hover:text-white/60"
        >
          <MoreHorizontal className="h-3 w-3" strokeWidth={2} />
        </button>
      </div>

      {/* Hand image — identical to dictionary cards */}
      <div className="aspect-square w-full flex items-center justify-center p-2">
        <img
          src={`/asl/${letter}.png`}
          alt={`ASL sign for letter ${letter}`}
          className="w-full h-full object-contain object-center select-none pointer-events-none
                     transition-[transform] duration-300 ease-out"
          style={{ transform: `scale(${scale})` }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLImageElement).style.transform = `scale(${scale * 1.03})`;
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLImageElement).style.transform = `scale(${scale})`;
          }}
          draggable={false}
        />
      </div>

      {/* Bottom: timestamp + accuracy */}
      <div className="flex items-center justify-between px-3 pb-3 pt-1">
        <p className="font-mono text-[9.5px] text-muted-foreground/70">{timestamp}</p>
        <p className="font-mono text-[9.5px] text-primary/70">{accuracy}%</p>
      </div>
    </div>
  );
};
