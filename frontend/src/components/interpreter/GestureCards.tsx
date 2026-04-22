import { Play } from "lucide-react";

/**
 * Two minimal cards for the dynamic motion letters J and Z.
 * Mesh-hand line icon + visual motion guide for each.
 */
export const GestureCards = () => {
  return (
    <div className="space-y-3">
      <p className="px-2 text-[10.5px] font-medium uppercase tracking-[0.22em] text-muted-foreground/80">
        Dynamic Gestures
      </p>

      {/* J */}
      <GestureCard
        letter="J"
        instruction={"Trace a 'J' shape\ndownward"}
        guide={
          <svg viewBox="0 0 80 60" fill="none" className="h-14 w-20">
            <defs>
              <linearGradient id="grad-j-side" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stopColor="hsl(339 100% 65%)" />
                <stop offset="1" stopColor="hsl(252 100% 68%)" />
              </linearGradient>
              <marker id="arr-j-side" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
                <path d="M0,0 L6,3 L0,6 Z" fill="hsl(252 100% 68%)" />
              </marker>
            </defs>

            {/* Mesh fist with index up */}
            <g stroke="hsl(0 0% 92% / 0.85)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" fill="none">
              {/* index finger up */}
              <path d="M22 32 V14" />
              {/* fist outline */}
              <path d="M14 34 Q14 28 18 27 L18 32 M18 32 Q18 27 22 27 L22 32 M22 32 Q22 28 26 28 L26 33 M26 33 Q26 30 30 30 L30 36 Q30 44 22 44 Q14 44 14 36 Z" />
              {/* knuckle dots */}
            </g>
            {/* tip dot */}
            <circle cx="22" cy="14" r="1.6" fill="hsl(0 0% 100%)" />

            {/* Curved arrow downward from tip */}
            <path
              d="M28 16 Q44 16 44 30 Q44 42 34 42 Q28 42 28 36"
              stroke="url(#grad-j-side)"
              strokeWidth="1.6"
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
          <svg viewBox="0 0 80 60" fill="none" className="h-14 w-20">
            <defs>
              <linearGradient id="grad-z-side" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stopColor="hsl(339 100% 65%)" />
                <stop offset="1" stopColor="hsl(252 100% 68%)" />
              </linearGradient>
              <marker id="arr-z-side" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
                <path d="M0,0 L6,3 L0,6 Z" fill="hsl(252 100% 68%)" />
              </marker>
            </defs>

            {/* Mesh fist (slightly tilted, index pointing) */}
            <g stroke="hsl(0 0% 92% / 0.85)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" fill="none">
              <path d="M14 38 Q14 30 22 30 Q30 30 30 38 Q30 46 22 46 Q14 46 14 38 Z" />
              <path d="M22 30 V20" />
              <path d="M22 33 H28" />
            </g>
            <circle cx="22" cy="20" r="1.6" fill="hsl(0 0% 100%)" />

            {/* Z path */}
            <path
              d="M40 14 H62 L40 40 H62"
              stroke="url(#grad-z-side)"
              strokeWidth="1.8"
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
  <div className="press glass glass-inner-highlight group rounded-2xl p-3 hover:bg-white/[0.06] hover:glow-pink-soft">
    <div className="mb-1 flex items-start justify-between">
      <span className="font-mono text-[12px] text-foreground/90">{letter}</span>
    </div>

    <div className="flex items-center justify-between gap-2">
      <div className="flex-1">{guide}</div>
      <button
        aria-label={`Play ${letter} demo`}
        className="press grid h-7 w-7 shrink-0 place-items-center rounded-full bg-white/[0.05] text-primary ring-1 ring-primary/30 hover:bg-primary/15 hover:glow-pink-soft"
      >
        <Play className="h-2.5 w-2.5 fill-current" strokeWidth={0} />
      </button>
    </div>

    <p className="mt-2 whitespace-pre-line text-center text-[10.5px] leading-snug text-muted-foreground">
      {instruction}
    </p>
  </div>
);
