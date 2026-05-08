// ── ConnectionStatus ──────────────────────────────────────────────────────────
// Replaces the fake ConfidenceRing with a real backend connection indicator.
// Shows a subtle floating badge driven by actual API reachability.

import { cn } from "@/lib/utils";
import type { ConnectionStatus as ConnectionStatusType } from "@/hooks/usePredictionLoop";

interface ConnectionStatusProps {
  status: ConnectionStatusType;
}

const STATUS_CONFIG: Record<
  ConnectionStatusType,
  { label: string; dotClass: string; textClass: string; pulse: boolean }
> = {
  connected: {
    label: "Connected",
    dotClass: "bg-emerald-400",
    textClass: "text-emerald-300",
    pulse: false,
  },
  reconnecting: {
    label: "Reconnecting…",
    dotClass: "bg-amber-400",
    textClass: "text-amber-300",
    pulse: true,
  },
  offline: {
    label: "Backend offline",
    dotClass: "bg-red-500",
    textClass: "text-red-400",
    pulse: false,
  },
};

export const ConnectionStatus = ({ status }: ConnectionStatusProps) => {
  const cfg = STATUS_CONFIG[status];

  return (
    <div
      className={cn(
        "animate-float-y flex flex-col items-center gap-3 rounded-2xl border border-white/[0.07] bg-black/50 px-4 py-4 backdrop-blur-md",
        "shadow-[inset_0_1px_0_hsl(0_0%_100%/0.07)]"
      )}
    >
      {/* Dot */}
      <span className="relative flex h-3 w-3">
        {cfg.pulse && (
          <span
            className={cn(
              "absolute inline-flex h-full w-full animate-ping rounded-full opacity-75",
              cfg.dotClass
            )}
          />
        )}
        <span
          className={cn(
            "relative inline-flex h-3 w-3 rounded-full",
            cfg.dotClass
          )}
        />
      </span>

      {/* Status label */}
      <span
        className={cn(
          "font-mono text-[9.5px] uppercase tracking-[0.22em]",
          cfg.textClass
        )}
      >
        {cfg.label}
      </span>

      <div className="h-px w-full bg-white/[0.06]" />

      {/* API tag */}
      <span className="font-mono text-[9px] uppercase tracking-[0.18em] text-muted-foreground/50">
        FastAPI
      </span>
    </div>
  );
};
