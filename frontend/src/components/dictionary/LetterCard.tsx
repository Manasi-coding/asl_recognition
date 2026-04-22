import { Play } from "lucide-react";
import { cn } from "@/lib/utils";
import { HandMesh } from "@/components/interpreter/HandMesh";

interface LetterCardProps {
  letter: string;
}

export const LetterCard = ({ letter }: LetterCardProps) => {
  const isJ = letter === "J";
  const isZ = letter === "Z";

  return (
    <div
      className={cn(
        "press glass glass-inner-highlight group relative overflow-hidden rounded-2xl p-3",
        "transition-shadow duration-300 hover:bg-white/[0.07] hover:glow-pink-soft"
      )}
    >
      {/* Top row: letter label + play button */}
      <div className="flex items-start justify-between">
        <span className="text-[15px] font-medium leading-none text-foreground">
          {letter}
        </span>
        <button
          aria-label={`Play ${letter} sign`}
          className="press grid h-6 w-6 place-items-center rounded-full bg-white/[0.05] text-muted-foreground opacity-80 ring-1 ring-white/10 transition-all hover:text-foreground hover:glow-pink-soft"
        >
          <Play className="h-2.5 w-2.5 fill-current" strokeWidth={0} />
        </button>
      </div>

      {/* Hand mesh visual */}
      <div className="relative mt-1 aspect-square w-full">
        <HandMesh letter={letter} showJPath={isJ} showZPath={isZ} />
      </div>
    </div>
  );
};
