// ── useHistoryStore ────────────────────────────────────────────────────────────
// Global singleton state. Sessions are ONLY created/ended by the Interpreter
// (START / STOP). Page navigation never touches session lifecycle.

import { useCallback, useEffect, useState } from "react";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface HistoryItem {
  id: string;
  letter: string;
  timestamp: string;
  accuracy: number;
  sessionId: number;
}

export interface Session {
  id: number;
  startedAt: Date;
  endedAt?: Date;
  duration: number;   // seconds, live-updated while active
  letters: HistoryItem[];
  active: boolean;
}

// ── Module-level singleton ─────────────────────────────────────────────────────
// Lives outside React so it survives tab switches without losing data.

let _nextId = 1;
let _sessions: Session[] = [];        // empty until first START press
let _listeners: Array<() => void> = [];
let _timerHandle: ReturnType<typeof setInterval> | undefined;

function notify() {
  _listeners.forEach((fn) => fn());
}

function subscribe(fn: () => void) {
  _listeners.push(fn);
  return () => { _listeners = _listeners.filter((l) => l !== fn); };
}

function activeSession(): Session | undefined {
  return _sessions.find((s) => s.active);
}

// ── Public mutators (called from Index.tsx) ───────────────────────────────────

/** Called ONLY when user presses START in the Interpreter. */
export function startNewSession() {
  // Safety: end any stale active session first
  _sessions.forEach((s) => { if (s.active) { s.active = false; s.endedAt = new Date(); } });

  const session: Session = {
    id: _nextId++,
    startedAt: new Date(),
    duration: 0,
    letters: [],
    active: true,
  };
  _sessions = [session, ..._sessions];

  // Start live timer — ticks only while session is active
  clearInterval(_timerHandle);
  _timerHandle = setInterval(() => {
    const s = activeSession();
    if (s) { s.duration += 1; notify(); }
    else    { clearInterval(_timerHandle); }   // auto-stop if no active session
  }, 1000);

  notify();
}

/** Called ONLY when user presses STOP in the Interpreter. */
export function endSession() {
  const s = activeSession();
  if (s) {
    s.active = false;
    s.endedAt = new Date();
  }
  clearInterval(_timerHandle);
  notify();
}

/**
 * Push a recognised letter into the active session.
 * Called from Index.tsx on each NEW prediction (consecutive duplicates from
 * holding a pose are already filtered by usePredictionLoop; we additionally
 * reset tracking when the hand leaves frame so L → pause → L is recorded twice).
 */
export function pushHistoryItem(letter: string, accuracy: number) {
  const s = activeSession();
  if (!s) return;                        // never record outside an active session

  const entry: HistoryItem = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    letter,
    accuracy,
    sessionId: s.id,
    timestamp: new Date().toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }),
  };

  s.letters = [entry, ...s.letters];    // newest first
  notify();
}

/** Clear ALL sessions and history (Clear History button). */
export function resetAllHistory() {
  _sessions = [];
  _nextId = 1;
  if (_timerHandle) clearInterval(_timerHandle);
  _timerHandle = undefined;
  notify();
}

/** Clear letters in the active session only. */
export function clearCurrentSession() {
  const s = activeSession();
  if (s) { s.letters = []; notify(); }
}

// ── Hook (used by History.tsx) ────────────────────────────────────────────────

export function useHistoryStore() {
  const [, forceUpdate] = useState(0);

  useEffect(() => {
    const unsub = subscribe(() => forceUpdate((n) => n + 1));
    return unsub;
  }, []);

  // ── Derived stats ──────────────────────────────────────────────────────────

  const allEntries = _sessions.flatMap((s) => s.letters);
  const totalLetters = allEntries.length;

  const avgAccuracy =
    allEntries.length > 0
      ? Math.round(allEntries.reduce((sum, e) => sum + e.accuracy, 0) / allEntries.length)
      : 0;

  const current = activeSession();
  const sessionLabel = current
    ? `#${String(current.id).padStart(2, "0")}`
    : _sessions.length > 0
    ? `#${String(_sessions[0].id).padStart(2, "0")} (ended)`
    : "—";

  const formatDuration = (sec: number) => {
    if (sec === 0) return "0s";
    if (sec < 60) return `${sec}s`;
    return `${Math.floor(sec / 60)}m ${String(sec % 60).padStart(2, "0")}s`;
  };

  const sessionDuration = formatDuration(current?.duration ?? _sessions[0]?.duration ?? 0);

  // Show letters from the most recent session (active or last ended)
  const recentItems: HistoryItem[] = _sessions[0]?.letters ?? [];

  const clearHistory = useCallback(() => { resetAllHistory(); }, []);

  return {
    totalLetters,
    sessionLabel,
    avgAccuracy,
    sessionDuration,
    recentItems,
    allEntries, // Expose for PDF export
    clearHistory,
    isSessionActive: !!current,
    sessions: _sessions,
  };
}
