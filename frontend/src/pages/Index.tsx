import { useEffect, useState } from "react";
import { Sidebar } from "@/components/interpreter/Sidebar";
import { CameraPanel } from "@/components/interpreter/CameraPanel";
import { ConfidenceRing } from "@/components/interpreter/ConfidenceRing";
import { ScrubBar } from "@/components/interpreter/ScrubBar";
import { Controls } from "@/components/interpreter/Controls";
import { Dictionary } from "@/pages/Dictionary";
import { History } from "@/pages/History";

const PHRASE = "HELLOJZ";

type NavId = "interpreter" | "dictionary" | "history";

const formatElapsed = (s: number) => {
  const hh = String(Math.floor(s / 3600)).padStart(2, "0");
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
};

const Index = () => {
  const [activeNav, setActiveNav] = useState<NavId>("interpreter");
  const [isRecording, setIsRecording] = useState(true);
  const [letters, setLetters] = useState<string[]>(["H", "E", "L", "L", "O", "J"]);
  const [confidence, setConfidence] = useState(95);
  const [elapsed, setElapsed] = useState(12);

  // Stream phrase letters
  useEffect(() => {
    if (!isRecording || activeNav !== "interpreter") return;
    const id = setInterval(() => {
      setLetters((prev) =>
        prev.length >= PHRASE.length ? prev : [...prev, PHRASE[prev.length]]
      );
      setConfidence(88 + Math.floor(Math.random() * 11));
    }, 1500);
    return () => clearInterval(id);
  }, [isRecording, activeNav]);

  // Tick session timer
  useEffect(() => {
    if (!isRecording || activeNav !== "interpreter") return;
    const id = setInterval(() => setElapsed((v) => v + 1), 1000);
    return () => clearInterval(id);
  }, [isRecording, activeNav]);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      <Sidebar active={activeNav} onChange={setActiveNav} />

      {activeNav === "dictionary" ? (
        <Dictionary />
      ) : activeNav === "history" ? (
        <History />
      ) : (
        <main className="relative flex flex-1 flex-col overflow-hidden p-6">
          {/* Outer rounded container framing the whole interpreter */}
          <div className="glass glass-inner-highlight relative flex flex-1 flex-col overflow-hidden rounded-[28px] p-6">
            {/* Top label */}
            <div className="flex items-center gap-2 px-1 pb-4">
              <span className="h-1.5 w-1.5 rounded-full bg-primary shadow-[0_0_8px_hsl(var(--primary))]" />
              <span className="font-mono text-[10.5px] uppercase tracking-[0.24em] text-muted-foreground">
                Live Interpreter
              </span>
            </div>

            {/* Camera + floating ring */}
            <section className="relative flex flex-1 items-center justify-center">
              <div className="relative w-full max-w-[1040px]">
                <CameraPanel />
                <div className="absolute right-6 top-1/2 -translate-y-1/2">
                  <ConfidenceRing value={confidence} />
                </div>
              </div>
            </section>

            {/* Scrub bar */}
            <div className="mt-5">
              <ScrubBar letters={letters} activeIndex={letters.length - 1} slots={8} />
            </div>
          </div>

          {/* Control bar — separate panel below */}
          <div className="mt-4">
            <Controls
              isRecording={isRecording}
              onToggleRecord={() => setIsRecording((v) => !v)}
              onPause={() => setIsRecording(false)}
              elapsed={formatElapsed(elapsed)}
            />
          </div>
        </main>
      )}
    </div>
  );
};

export default Index;
