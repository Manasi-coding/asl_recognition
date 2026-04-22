import { Type, CalendarDays, Zap, Clock, type LucideIcon } from "lucide-react";

interface Stat {
  icon: LucideIcon;
  label: string;
  value: string;
  sub: string;
}

const STATS: Stat[] = [
  { icon: Type, label: "Total Letters", value: "126", sub: "All time" },
  { icon: CalendarDays, label: "Today", value: "34", sub: "Recognized" },
  { icon: Zap, label: "Accuracy", value: "95%", sub: "Average" },
  { icon: Clock, label: "Last Session", value: "2m 48s", sub: "Duration" },
];

export const StatsCards = () => {
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
          <div className="min-w-0 leading-tight">
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              {label}
            </p>
            <p className="mt-1 text-[20px] font-medium tracking-tight text-foreground">
              {value}
            </p>
            <p className="mt-0.5 text-[11px] text-muted-foreground">{sub}</p>
          </div>
        </div>
      ))}
    </div>
  );
};
