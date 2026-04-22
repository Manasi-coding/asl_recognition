import { Search, SlidersHorizontal, Upload, History as HistoryIcon } from "lucide-react";

interface HistoryHeaderProps {
  query: string;
  onQueryChange: (v: string) => void;
}

export const HistoryHeader = ({ query, onQueryChange }: HistoryHeaderProps) => {
  return (
    <header className="flex flex-col gap-4 px-10 pt-8 md:flex-row md:items-center md:justify-between">
      <div className="flex items-center gap-3.5">
        <div className="glass glass-inner-highlight grid h-11 w-11 place-items-center rounded-2xl">
          <HistoryIcon className="h-[18px] w-[18px] text-primary" strokeWidth={1.75} />
        </div>
        <div>
          <p className="font-mono text-[11px] font-medium uppercase tracking-[0.22em] text-foreground">
            History
          </p>
          <p className="mt-1 text-[12.5px] text-muted-foreground">
            Review your recently recognized letters
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {/* Search */}
        <div className="glass glass-inner-highlight flex h-10 items-center gap-2 rounded-full px-4">
          <Search className="h-3.5 w-3.5 text-muted-foreground" strokeWidth={1.75} />
          <input
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder="Search letters..."
            className="w-44 bg-transparent text-[12.5px] text-foreground placeholder:text-muted-foreground focus:outline-none"
          />
        </div>

        {/* Filter */}
        <button className="press glass glass-inner-highlight flex h-10 items-center gap-2 rounded-full px-4 text-[12.5px] text-foreground/90 hover:glow-pink-soft">
          <SlidersHorizontal className="h-3.5 w-3.5" strokeWidth={1.75} />
          Filter
        </button>

        {/* Export — highlighted with pink glow */}
        <button
          className="press flex h-10 items-center gap-2 rounded-full px-4 text-[12.5px] font-medium text-foreground ring-1 ring-primary/50 glow-pink-soft hover:glow-pink"
          style={{ background: "hsl(339 100% 65% / 0.08)" }}
        >
          <Upload className="h-3.5 w-3.5 text-primary" strokeWidth={1.75} />
          Export
        </button>
      </div>
    </header>
  );
};
