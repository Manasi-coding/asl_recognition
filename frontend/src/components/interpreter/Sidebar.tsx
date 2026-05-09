import { Hand, BookText, History } from "lucide-react";
import { cn } from "@/lib/utils";
import { GestureCards } from "./GestureCards";

type NavId = "interpreter" | "dictionary" | "history";

const navItems: { id: NavId; label: string; icon: React.ComponentType<React.SVGProps<SVGSVGElement>> }[] = [
  { id: "interpreter", label: "Interpreter", icon: Hand },
  { id: "dictionary", label: "Dictionary", icon: BookText },
  { id: "history", label: "History", icon: History },
];

interface SidebarProps {
  active: NavId;
  onChange: (id: NavId) => void;
}

export const Sidebar = ({ active, onChange }: SidebarProps) => {
  return (
    <aside className="flex h-full w-60 shrink-0 flex-col gap-6 border-r border-border bg-background px-4 py-6">
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-2">
        <div className="relative grid h-9 w-9 place-items-center rounded-xl"
          style={{ background: "var(--gradient-pink-violet)" }}>
          <span className="text-base font-semibold text-primary-foreground">S</span>
          <span aria-hidden className="absolute inset-0 rounded-xl ring-1 ring-inset ring-white/15" />
        </div>
        <p className="text-[16px] font-semibold tracking-tight text-foreground">Signify</p>
      </div>

      {/* Nav */}
      <nav className="flex flex-col gap-1">
        {navItems.map(({ id, label, icon: Icon }) => {
          const isActive = active === id;
          return (
            <button
              key={id}
              onClick={() => onChange(id)}
              className={cn(
                "press group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-[13.5px] font-medium",
                "text-muted-foreground hover:text-foreground",
                isActive
                  ? "bg-white/[0.05] text-foreground glow-pink-soft"
                  : "hover:bg-white/[0.03]"
              )}
            >
              {isActive && (
                <span
                  aria-hidden
                  className="absolute left-0 top-1/2 h-5 w-[2px] -translate-y-1/2 rounded-full bg-primary shadow-[0_0_10px_hsl(var(--primary))]"
                />
              )}
              <Icon
                className={cn(
                  "h-[17px] w-[17px] transition-colors",
                  isActive ? "text-primary" : "text-muted-foreground group-hover:text-foreground"
                )}
                strokeWidth={1.75}
              />
              <span>{label}</span>
            </button>
          );
        })}
      </nav>

      {/* Dynamic Gestures */}
      <GestureCards />

    </aside>
  );
};
