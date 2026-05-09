import { useMemo, useState, useEffect } from "react";
import { HistoryHeader } from "@/components/history/HistoryHeader";
import { StatsCards } from "@/components/history/StatsCards";
import { RecentLettersSection } from "@/components/history/RecentLettersSection";
import { Pagination } from "@/components/history/Pagination";
import { useHistoryStore } from "@/hooks/useHistoryStore";

const PAGE_SIZE = 24; // 4 rows × 6 cols

export const History = () => {
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);

  const {
    totalLetters,
    allEntries,
    sessions,
    clearHistory,
  } = useHistoryStore();

  const [selectedSessionId, setSelectedSessionId] = useState<number | null>(null);

  // Set initial selected session to most recent
  useEffect(() => {
    if (sessions.length > 0 && selectedSessionId === null) {
      setSelectedSessionId(sessions[0].id);
    }
  }, [sessions, selectedSessionId]);

  const selectedSession = useMemo(() =>
    sessions.find(s => s.id === selectedSessionId) || sessions[0],
    [sessions, selectedSessionId]
  );

  const displayItems = selectedSession?.letters ?? [];

  const displayAccuracy = useMemo(() => {
    if (!selectedSession || selectedSession.letters.length === 0) return 0;
    return Math.round(selectedSession.letters.reduce((sum, e) => sum + e.accuracy, 0) / selectedSession.letters.length);
  }, [selectedSession]);

  const formatDuration = (sec: number) => {
    if (sec === 0) return "0s";
    if (sec < 60) return `${sec}s`;
    return `${Math.floor(sec / 60)}m ${String(sec % 60).padStart(2, "0")}s`;
  };

  const displayDuration = useMemo(() => formatDuration(selectedSession?.duration ?? 0), [selectedSession]);
  const sessionLabel = selectedSession
    ? `#${String(selectedSession.id).padStart(2, "0")}${selectedSession.active ? "" : " (ended)"}`
    : "—";

  // Filter by query
  const filtered = useMemo(() => {
    const q = query.trim().toUpperCase();
    return q ? displayItems.filter((i) => i.letter.includes(q)) : displayItems;
  }, [query, displayItems]);

  // Paginate
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  // Reset to page 1 whenever filter or session changes
  useEffect(() => setPage(1), [query, selectedSessionId]);

  return (
    <main className="relative flex h-full flex-1 flex-col overflow-hidden">
      <HistoryHeader
        query={query}
        onQueryChange={setQuery}
        stats={{
          totalLetters,
          sessionLabel,
          avgAccuracy: displayAccuracy,
          sessionDuration: displayDuration,
          allEntries: displayItems // Export selected session letters
        }}
      />

      <section className="flex-1 overflow-y-auto px-10 pb-8 pt-6">
        <StatsCards
          totalLetters={totalLetters}
          sessionLabel={sessionLabel}
          avgAccuracy={displayAccuracy}
          sessionDuration={displayDuration}
          sessions={sessions}
          selectedSessionId={selectedSessionId}
          onSessionChange={setSelectedSessionId}
        />

        <RecentLettersSection
          items={paginated}
          onClear={clearHistory}
        />

        {totalPages > 1 && (
          <Pagination page={page} totalPages={totalPages} onChange={setPage} />
        )}
      </section>
    </main>
  );
};

export default History;
