import { Type, Hash, Zap, Clock, type LucideIcon } from "lucide-react";
import { type Session } from "@/hooks/useHistoryStore";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface Stat {
  icon: LucideIcon;
  label: string;
  value: string;
  sub: string;
}

interface StatsCardsProps {
  totalLetters: number;
  sessionLabel: string;
  avgAccuracy: number;
  sessionDuration: string;
  sessions: Session[];
  selectedSessionId: number | null;
  onSessionChange: (id: number) => void;
}

export const StatsCards = ({
  totalLetters,
  sessionLabel,
  avgAccuracy,
  sessionDuration,
  sessions,
  selectedSessionId,
  onSessionChange,
}: StatsCardsProps) => {
  const STATS: Stat[] = [
    { icon: Type,  label: "Total Letters",    value: String(totalLetters),          sub: "All sessions"     },
    { icon: Hash,  label: "Current Session",  value: sessionLabel,                  sub: "Select session"   },
    { icon: Zap,   label: "Accuracy",         value: avgAccuracy > 0 ? `${avgAccuracy}%` : "—", sub: "Average" },
    { icon: Clock, label: "Session Time",     value: sessionDuration,               sub: "Duration"  },
  ];

  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
      {STATS.map(({ icon: Icon, label, value, sub }) => (
        <div
          key={label}
          className="press glass glass-inner-highlight flex items-center gap-3.5 rounded-2xl p-4 hover:glow-pink-soft"
        >
          <div
            className="grid h-11 w-11 shrink-0 place-items-center rounded-xl ring-1 ring-primary/30"
            style={{ background: "hsl(339 100% 65% / 0.08)" }}
          >
            <Icon className="h-[18px] w-[18px] text-primary" strokeWidth={1.75} />
          </div>
          <div className="min-w-0 flex-1 leading-tight">
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              {label}
            </p>
            {label === "Current Session" ? (
              <Select
                value={selectedSessionId ? String(selectedSessionId) : undefined}
                onValueChange={(v) => onSessionChange(Number(v))}
              >
                <SelectTrigger className="mt-1 h-auto border-none bg-transparent p-0 text-[20px] font-medium tracking-tight text-foreground shadow-none focus:ring-0 [&>svg]:text-primary [&>svg]:opacity-100">
                  <SelectValue placeholder="Session" />
                </SelectTrigger>
                <SelectContent className="border-white/10 bg-black/95 backdrop-blur-xl">
                  {sessions.map((s) => (
                    <SelectItem
                      key={s.id}
                      value={String(s.id)}
                      className="font-mono text-[13px] text-foreground focus:bg-white/5 focus:text-primary"
                    >
                      Session #{String(s.id).padStart(2, "0")}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <p className="mt-1 text-[20px] font-medium tracking-tight text-foreground">
                {value}
              </p>
            )}
            <p className="mt-0.5 text-[11px] text-muted-foreground">{sub}</p>
          </div>
        </div>
      ))}
    </div>
  );
};
