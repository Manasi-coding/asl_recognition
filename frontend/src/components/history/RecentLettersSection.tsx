import { Trash2 } from "lucide-react";
import { HistoryLetterCard } from "./HistoryLetterCard";
import type { HistoryItem } from "@/hooks/useHistoryStore";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";

interface RecentLettersSectionProps {
  items: HistoryItem[];
  onClear: () => void;
}

export const RecentLettersSection = ({ items, onClear }: RecentLettersSectionProps) => {
  return (
    <section className="mt-6">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-primary shadow-[0_0_10px_hsl(var(--primary))]" />
          <h2 className="text-[14px] font-medium text-foreground">Recent Letters</h2>
          {items.length > 0 && (
            <span className="font-mono text-[11px] text-muted-foreground/60">
              ({items.length})
            </span>
          )}
        </div>

        <AlertDialog>
          <AlertDialogTrigger asChild>
            <button
              className="press glass glass-inner-highlight flex items-center gap-2 rounded-full
                         px-3.5 py-1.5 text-[11.5px] text-foreground/90 hover:glow-pink-soft"
            >
              <Trash2 className="h-3 w-3" strokeWidth={1.75} />
              Clear History
            </button>
          </AlertDialogTrigger>
          <AlertDialogContent className="border-white/10 bg-black/95 backdrop-blur-xl">
            <AlertDialogHeader>
              <AlertDialogTitle className="font-mono text-[14px] font-medium uppercase tracking-widest text-foreground">
                Clear History
              </AlertDialogTitle>
              <AlertDialogDescription className="text-[13px] text-muted-foreground">
                Clear all history permanently?
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter className="mt-4">
              <AlertDialogCancel className="rounded-full border-white/5 bg-white/5 text-[12px] hover:bg-white/10 hover:text-white">
                Cancel
              </AlertDialogCancel>
              <AlertDialogAction
                onClick={onClear}
                className="rounded-full bg-primary text-[12px] font-medium text-primary-foreground hover:bg-primary/90 hover:glow-pink"
              >
                Clear History
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>

      {items.length === 0 ? (
        <div className="flex h-64 flex-col items-center justify-center opacity-40">
          <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-muted-foreground">
            No interpretation history yet
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 md:grid-cols-6">
          {items.map((item) => (
            <HistoryLetterCard key={item.id} item={item} />
          ))}
        </div>
      )}
    </section>
  );
};
