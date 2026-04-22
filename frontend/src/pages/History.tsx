import { useMemo, useState } from "react";
import { HistoryHeader } from "@/components/history/HistoryHeader";
import { StatsCards } from "@/components/history/StatsCards";
import { RecentLettersSection, type RecentLetter } from "@/components/history/RecentLettersSection";
import { Pagination } from "@/components/history/Pagination";

// Mock recent letter data — matches reference layout (3 rows × 5 cols)
const RECENT: RecentLetter[] = [
  { letter: "H", timestamp: "10:24:31 AM" },
  { letter: "E", timestamp: "10:24:28 AM" },
  { letter: "L", timestamp: "10:24:25 AM" },
  { letter: "L", timestamp: "10:24:22 AM" },
  { letter: "O", timestamp: "10:24:18 AM" },
  { letter: "J", timestamp: "10:24:12 AM" },
  { letter: "A", timestamp: "10:24:08 AM" },
  { letter: "M", timestamp: "10:24:05 AM" },
  { letter: "I", timestamp: "10:24:02 AM" },
  { letter: "L", timestamp: "10:24:59 AM" },
  { letter: "Y", timestamp: "10:23:55 AM" },
  { letter: "O", timestamp: "10:23:51 AM" },
  { letter: "U", timestamp: "10:23:48 AM" },
  { letter: "?", timestamp: "10:23:45 AM" },
  { letter: "N", timestamp: "10:23:41 AM" },
];

export const History = () => {
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);

  const filtered = useMemo(() => {
    const q = query.trim().toUpperCase();
    if (!q) return RECENT;
    return RECENT.filter((r) => r.letter.includes(q));
  }, [query]);

  return (
    <main className="relative flex h-full flex-1 flex-col overflow-hidden">
      <HistoryHeader query={query} onQueryChange={setQuery} />

      <section className="flex-1 overflow-y-auto px-10 pb-8 pt-6">
        <StatsCards />
        <RecentLettersSection letters={filtered} />
        <Pagination page={page} totalPages={5} onChange={setPage} />
      </section>
    </main>
  );
};

export default History;
