import { useMemo, useState } from "react";
import { DictionaryHeader } from "@/components/dictionary/DictionaryHeader";
import { LetterGrid } from "@/components/dictionary/LetterGrid";

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

export const Dictionary = () => {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toUpperCase();
    if (!q) return ALPHABET;
    return ALPHABET.filter((l) => l.includes(q));
  }, [query]);

  return (
    <main className="relative flex h-full flex-1 flex-col overflow-hidden">
      <DictionaryHeader query={query} onQueryChange={setQuery} />

      <section className="flex-1 overflow-y-auto px-10 pb-10 pt-6">
        <LetterGrid letters={filtered} />
      </section>
    </main>
  );
};

export default Dictionary;
