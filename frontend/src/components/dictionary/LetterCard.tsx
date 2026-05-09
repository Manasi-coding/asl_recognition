import { cn } from "@/lib/utils";

// Letters whose illustrations still touch card edges — scale them down slightly
const SCALE_OVERRIDES: Record<string, number> = {
  B: 0.88, F: 0.88, K: 0.88, L: 0.88, U: 0.88, V: 0.88, W: 0.88,
};

interface LetterCardProps {
  letter: string;
}

export const LetterCard = ({ letter }: LetterCardProps) => {
  const scale = SCALE_OVERRIDES[letter] ?? 1;
  return (
    <div
      className={cn(
        "group relative flex flex-col overflow-hidden rounded-xl",
        "bg-black border border-white/[0.07]",
        "transition-all duration-300",
        "hover:border-primary/25",
        "hover:shadow-[0_0_16px_rgba(255,45,140,0.12)]"
      )}
    >
      {/* Letter label — top-left only */}
      <div className="px-3 pt-3 pb-0">
        <span className="font-mono text-[11px] font-semibold tracking-[0.2em] text-white/60 uppercase">
          {letter}
        </span>
      </div>

      {/* Hand illustration — square, centred, no inner glow frame */}
      <div className="aspect-square w-full flex items-center justify-center p-2">
        <img
          src={`/asl/${letter}.png`}
          alt={`ASL sign for letter ${letter}`}
          className={cn(
            "w-full h-full object-contain object-center select-none pointer-events-none",
            "transition-[transform] duration-300 ease-out"
          )}
          style={{ transform: `scale(${scale})` }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLImageElement).style.transform = `scale(${scale * 1.03})`; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLImageElement).style.transform = `scale(${scale})`; }}
          draggable={false}
        />
      </div>
    </div>
  );
};