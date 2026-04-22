import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface PaginationProps {
  page: number;
  totalPages: number;
  onChange: (page: number) => void;
}

export const Pagination = ({ page, totalPages, onChange }: PaginationProps) => {
  const pages = Array.from({ length: totalPages }, (_, i) => i + 1);

  const btn =
    "press grid h-9 w-9 place-items-center rounded-full text-[12px] text-muted-foreground hover:text-foreground";

  return (
    <nav className="mt-8 flex items-center justify-center gap-1.5" aria-label="Pagination">
      <button
        onClick={() => onChange(Math.max(1, page - 1))}
        disabled={page === 1}
        className={cn(btn, "glass glass-inner-highlight disabled:opacity-40")}
        aria-label="Previous page"
      >
        <ChevronLeft className="h-3.5 w-3.5" strokeWidth={2} />
      </button>

      {pages.map((p) => {
        const isActive = p === page;
        return (
          <button
            key={p}
            onClick={() => onChange(p)}
            aria-current={isActive ? "page" : undefined}
            className={cn(
              btn,
              isActive
                ? "text-foreground ring-1 ring-primary/60 glow-pink-soft"
                : "hover:bg-white/[0.04]"
            )}
            style={
              isActive
                ? { background: "hsl(339 100% 65% / 0.10)" }
                : undefined
            }
          >
            {p}
          </button>
        );
      })}

      <button
        onClick={() => onChange(Math.min(totalPages, page + 1))}
        disabled={page === totalPages}
        className={cn(btn, "glass glass-inner-highlight disabled:opacity-40")}
        aria-label="Next page"
      >
        <ChevronRight className="h-3.5 w-3.5" strokeWidth={2} />
      </button>
    </nav>
  );
};
