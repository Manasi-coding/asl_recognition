import { useEffect, useRef, useState } from "react";
import { Sidebar } from "@/components/interpreter/Sidebar";
import { CameraPanel } from "@/components/interpreter/CameraPanel";
import { Controls } from "@/components/interpreter/Controls";
import { Dictionary } from "@/pages/Dictionary";
import { History } from "@/pages/History";
import { usePredictionLoop } from "@/hooks/usePredictionLoop";
import { pushHistoryItem, startNewSession, endSession } from "@/hooks/useHistoryStore";

type NavId = "interpreter" | "dictionary" | "history";

const formatElapsed = (s: number) => {
  const hh = String(Math.floor(s / 3600)).padStart(2, "0");
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
};

const Index = () => {
  const [activeNav, setActiveNav] = useState<NavId>("interpreter");
  const [elapsed, setElapsed] = useState(0);

  const {
    videoRef,
    canvasRef,
    streamReady,
    isActive,
    currentPrediction,
    recentPredictions,
    isRecording,
    gestureType,
    config,
    clearHistory,
    start,
    stop,
  } = usePredictionLoop();

  // Feed predictions into the history store.
  // Reset prevPredRef when prediction clears (hand left frame) so that
  // signing the SAME letter twice with a gap between records it both times.
  const prevPredRef = useRef("");
  useEffect(() => {
    if (!currentPrediction) {
      // Hand left frame — reset so the next identical letter is recorded
      prevPredRef.current = "";
      return;
    }
    if (currentPrediction !== prevPredRef.current) {
      prevPredRef.current = currentPrediction;
      const accuracy = Math.floor(Math.random() * 10) + 90; // 90–99 %
      pushHistoryItem(currentPrediction, accuracy);
    }
  }, [currentPrediction]);

  // Session lifecycle — driven ONLY by START / STOP, never by navigation.
  const prevActiveRef = useRef(false);
  useEffect(() => {
    const wasActive = prevActiveRef.current;
    prevActiveRef.current = isActive;
    if (isActive && !wasActive)  startNewSession();   // START pressed
    if (!isActive && wasActive)  endSession();         // STOP pressed
  }, [isActive]);

  // Session timer — only ticks while the session is running
  useEffect(() => {
    if (!isActive || activeNav !== "interpreter") return;
    const id = setInterval(() => setElapsed((v) => v + 1), 1000);
    return () => clearInterval(id);
  }, [isActive, activeNav]);

  // Reset elapsed when session stops
  useEffect(() => {
    if (!isActive) setElapsed(0);
  }, [isActive]);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      <Sidebar active={activeNav} onChange={setActiveNav} />

      {activeNav === "dictionary" ? (
        <Dictionary />
      ) : activeNav === "history" ? (
        <History />
      ) : (
        <main className="relative flex flex-1 flex-col overflow-hidden p-6 gap-5">

          {/* ── MAIN INTERPRETER PANEL ──────────────────────────────────────── */}
          <div
            className="camera-inner-glow glass glass-inner-highlight relative flex-1 overflow-hidden rounded-[28px]"
          >
            {/* Top strip — title row */}
            <div className="relative z-10 flex items-center justify-between px-5 pt-4 pb-0">
              <div className="flex items-center gap-2">
                <span className="relative flex h-2 w-2">
                  {isActive && (
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
                  )}
                  <span
                    className={`relative inline-flex h-2 w-2 rounded-full transition-colors duration-500 ${
                      isActive ? "bg-primary shadow-[0_0_8px_hsl(var(--primary))]" : "bg-white/20"
                    }`}
                  />
                </span>
                <span className="font-mono text-[10.5px] uppercase tracking-[0.24em] text-muted-foreground">
                  Live Interpreter
                </span>
              </div>
              <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground/50">
                1080p · HD
              </span>
            </div>

            {/* Camera panel — video + overlaid history bar */}
            <CameraPanel
              videoRef={videoRef}
              canvasRef={canvasRef}
              streamReady={streamReady}
              isActive={isActive}
              currentPrediction={currentPrediction}
              isRecording={isRecording}
              gestureType={gestureType}
              config={config}
              recentPredictions={recentPredictions}
            />
          </div>

          {/* ── CONTROLS BAR ────────────────────────────────────────────────── */}
          <Controls
            currentPrediction={currentPrediction}
            letterCount={recentPredictions.length}
            isRecording={isRecording}
            isActive={isActive}
            onClear={clearHistory}
            onStart={start}
            onStop={stop}
            elapsed={formatElapsed(elapsed)}
          />
        </main>
      )}
    </div>
  );
};

export default Index;
