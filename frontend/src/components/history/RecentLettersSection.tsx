import { Trash2 } from "lucide-react";
import { HistoryLetterCard } from "./HistoryLetterCard";

export interface RecentLetter {
  letter: string;
  timestamp: string;
}

interface RecentLettersSectionProps {
  letters: RecentLetter[];
  onClear?: () => void;
}

export const RecentLettersSection = ({ letters, onClear }: RecentLettersSectionProps) => {
  return (
    <section className="mt-6">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-primary shadow-[0_0_10px_hsl(var(--primary))]" />
          <h2 className="text-[14px] font-medium text-foreground">Recent Letters</h2>
        </div>
        <button
          onClick={onClear}
          className="press glass glass-inner-highlight flex items-center gap-2 rounded-full px-3.5 py-1.5 text-[11.5px] text-foreground/90 hover:glow-pink-soft"
        >
          <Trash2 className="h-3 w-3" strokeWidth={1.75} />
          Clear History
        </button>
      </div>

      <div className="grid grid-cols-3 gap-4 sm:grid-cols-4 md:grid-cols-5">
        {letters.map((l, i) => (
          <HistoryLetterCard key={`${l.letter}-${i}`} letter={l.letter} timestamp={l.timestamp} />
        ))}
      </div>
    </section>
  );
};
