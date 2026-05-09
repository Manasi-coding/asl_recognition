import { Search, Upload, History as HistoryIcon } from "lucide-react";
import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

interface HistoryHeaderProps {
  query: string;
  onQueryChange: (v: string) => void;
  stats: {
    totalLetters: number;
    sessionLabel: string;
    avgAccuracy: number;
    sessionDuration: string;
    allEntries: any[];
  };
}

export const HistoryHeader = ({ query, onQueryChange, stats }: HistoryHeaderProps) => {
  const handleExport = () => {
    const doc = new jsPDF();
    
    // Title
    doc.setFont("helvetica", "bold");
    doc.setFontSize(22);
    doc.text("Signify History Report", 105, 20, { align: "center" });
    
    // Summary Section
    doc.setFontSize(12);
    doc.setFont("helvetica", "normal");
    const summaryY = 40;
    doc.text(`Total Letters: ${stats.totalLetters}`, 20, summaryY);
    doc.text(`Current Session: ${stats.sessionLabel}`, 20, summaryY + 7);
    doc.text(`Accuracy: ${stats.avgAccuracy}%`, 20, summaryY + 14);
    doc.text(`Session Time: ${stats.sessionDuration}`, 20, summaryY + 21);
    
    // Calculate Frequencies
    const freqMap: Record<string, number> = {};
    stats.allEntries.forEach((entry) => {
      freqMap[entry.letter] = (freqMap[entry.letter] || 0) + 1;
    });
    
    const tableData = Object.entries(freqMap)
      .sort((a, b) => b[1] - a[1]) // Sort by frequency descending
      .map(([letter, freq]) => [letter, freq]);
      
    // Recent Letters Table
    autoTable(doc, {
      startY: summaryY + 35,
      head: [["Letter", "Frequency"]],
      body: tableData,
      theme: "grid",
      headStyles: { fillColor: [0, 0, 0], textColor: [255, 255, 255], fontStyle: "bold" },
      styles: { font: "helvetica", fontSize: 10 },
      margin: { left: 20, right: 20 },
    });
    
    doc.save("signify-history-report.pdf");
  };

  return (
    <header className="flex flex-col gap-4 px-10 pt-8 md:flex-row md:items-center md:justify-between">
      <div className="flex items-center gap-3.5">
        <div className="glass glass-inner-highlight grid h-11 w-11 place-items-center rounded-2xl">
          <HistoryIcon className="h-[18px] w-[18px] text-primary" strokeWidth={1.75} />
        </div>
        <div>
          <p className="font-mono text-[11px] font-medium uppercase tracking-[0.22em] text-foreground">
            History
          </p>
          <p className="mt-1 text-[12.5px] text-muted-foreground">
            Review your recently recognized letters
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {/* Search */}
        <div className="glass glass-inner-highlight flex h-10 items-center gap-2 rounded-full px-4">
          <Search className="h-3.5 w-3.5 text-muted-foreground" strokeWidth={1.75} />
          <input
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder="Search letters..."
            className="w-44 bg-transparent text-[12.5px] text-foreground placeholder:text-muted-foreground focus:outline-none"
          />
        </div>


        {/* Export — highlighted with pink glow */}
        <button
          onClick={handleExport}
          className="press flex h-10 items-center gap-2 rounded-full px-4 text-[12.5px] font-medium text-foreground ring-1 ring-primary/50 glow-pink-soft hover:glow-pink"
          style={{ background: "hsl(339 100% 65% / 0.08)" }}
        >
          <Upload className="h-3.5 w-3.5 text-primary" strokeWidth={1.75} />
          Export
        </button>
      </div>
    </header>
  );
};
