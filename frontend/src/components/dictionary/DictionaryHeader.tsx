import { Search, SlidersHorizontal } from "lucide-react";

interface DictionaryHeaderProps {
  query: string;
  onQueryChange: (v: string) => void;
}

export const DictionaryHeader = ({ query, onQueryChange }: DictionaryHeaderProps) => {
  return (
    <header className="flex flex-col gap-4 px-10 pt-8 md:flex-row md:items-end md:justify-between">
      <div>
        <p className="font-mono text-[10.5px] uppercase tracking-[0.22em] text-muted-foreground">
          Dictionary
        </p>
        <h1 className="mt-1 text-[22px] font-medium tracking-tight text-foreground">
          Browse and learn all sign language letters
        </h1>
      </div>

      <div className="flex items-center gap-2">
        {/* Search */}
        <div className="glass glass-inner-highlight flex h-10 items-center gap-2 rounded-full px-4">
          <Search className="h-3.5 w-3.5 text-muted-foreground" strokeWidth={1.75} />
          <input
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder="Search letters..."
            className="w-48 bg-transparent text-[12.5px] text-foreground placeholder:text-muted-foreground focus:outline-none"
          />
        </div>

        {/* Filter */}
        <button className="press glass glass-inner-highlight flex h-10 items-center gap-2 rounded-full px-4 text-[12.5px] text-foreground/90 hover:glow-pink-soft">
          <SlidersHorizontal className="h-3.5 w-3.5" strokeWidth={1.75} />
          Filter
        </button>
      </div>
    </header>
  );
};
