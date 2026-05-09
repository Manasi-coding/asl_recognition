import { cn } from "@/lib/utils";

/**
 * Two minimal cards for the dynamic motion letters J and Z.
 * Adapted to match the Dictionary page aesthetic while keeping motion guides.
 */
export const GestureCards = () => {
  return (
    <div className="space-y-2.5">
      <p className="px-2 text-[10.5px] font-medium uppercase tracking-[0.22em] text-muted-foreground/80">
        Dynamic Gestures
      </p>

      {/* J */}
      <GestureCard
        letter="J"
        instruction={"Trace a 'J' shape\ndownward"}
        guide={
          <svg viewBox="0 0 80 60" fill="none" className="h-full w-full">
            <defs>
              <linearGradient id="grad-j-side" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stopColor="hsl(339 100% 65%)" />
                <stop offset="1" stopColor="hsl(252 100% 68%)" />
              </linearGradient>
              <marker id="arr-j-side" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
                <path d="M0,0 L6,3 L0,6 Z" fill="hsl(252 100% 68%)" />
              </marker>
            </defs>
            <path
              d="M26 10 Q54 10 54 30 Q54 50 40 50 Q26 50 26 40"
              stroke="url(#grad-j-side)"
              strokeWidth="2.2"
              strokeLinecap="round"
              fill="none"
              markerEnd="url(#arr-j-side)"
            />
          </svg>
        }
      />

      {/* Z */}
      <GestureCard
        letter="Z"
        instruction={"Trace a 'Z' shape\nin the air"}
        guide={
          <svg viewBox="0 0 80 60" fill="none" className="h-full w-full">
            <defs>
              <linearGradient id="grad-z-side" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stopColor="hsl(339 100% 65%)" />
                <stop offset="1" stopColor="hsl(252 100% 68%)" />
              </linearGradient>
              <marker id="arr-z-side" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
                <path d="M0,0 L6,3 L0,6 Z" fill="hsl(252 100% 68%)" />
              </marker>
            </defs>
            <path
              d="M20 16 H60 L20 44 H60"
              stroke="url(#grad-z-side)"
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
              markerEnd="url(#arr-z-side)"
            />
          </svg>
        }
      />
    </div>
  );
};

interface GestureCardProps {
  letter: string;
  instruction: string;
  guide: React.ReactNode;
}

const GestureCard = ({ letter, instruction, guide }: GestureCardProps) => (
  <div
    className={cn(
      "group relative flex flex-col overflow-hidden rounded-xl",
      "bg-black border border-white/[0.07]",
      "transition-all duration-300",
      "hover:border-primary/25",
      "hover:shadow-[0_0_16px_rgba(255,45,140,0.12)]"
    )}
  >
    {/* Letter label */}
    <div className="px-3 pt-2 pb-0">
      <span className="font-mono text-[10px] font-semibold tracking-[0.2em] text-white/50 uppercase">
        {letter}
      </span>
    </div>

    {/* Split Layout: Left (Arrow) | Right (Hand) */}
    <div className="relative h-20 w-full flex items-center justify-between px-2.5 py-1">
      {/* LEFT: Motion Indicator (40% width) */}
      <div className="w-[40%] h-full flex items-center justify-center opacity-80 group-hover:opacity-100 transition-opacity duration-300">
        <div className="w-full h-full max-h-[48px] flex items-center justify-center">
          {guide}
        </div>
      </div>

      {/* RIGHT: Hand Illustration (60% width) */}
      <div className="w-[60%] h-full flex items-center justify-center overflow-visible">
        <img
          src={`/asl/${letter}.png`}
          alt={`ASL sign for letter ${letter}`}
          className={cn(
            "h-[115%] w-auto object-contain select-none pointer-events-none translate-x-1",
            "transition-transform duration-300 ease-out"
          )}
          onMouseEnter={(e) => { (e.currentTarget as HTMLImageElement).style.transform = "scale(1.05) translateX(4px)"; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLImageElement).style.transform = "scale(1) translateX(4px)"; }}
          draggable={false}
        />
      </div>
    </div>

    <p className="px-3 pb-2.5 text-center text-[9.5px] leading-tight text-muted-foreground whitespace-pre-line">
      {instruction}
    </p>
  </div>
);

