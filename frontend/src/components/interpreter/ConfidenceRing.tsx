interface ConfidenceRingProps {
  value: number; // 0-100
}

export const ConfidenceRing = ({ value }: ConfidenceRingProps) => {
  const size = 120;
  const stroke = 7;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const offset = c - (Math.max(0, Math.min(100, value)) / 100) * c;

  return (
    <div className="relative animate-float-y" style={{ width: size, height: size }}>
      {/* soft outer glow */}
      <div
        aria-hidden
        className="absolute inset-0 rounded-full"
        style={{
          background:
            "radial-gradient(circle, hsl(339 100% 65% / 0.18), hsl(252 100% 68% / 0.10) 50%, transparent 75%)",
          filter: "blur(8px)",
        }}
      />

      <svg width={size} height={size} className="-rotate-90">
        <defs>
          <linearGradient id="ring-pv" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="hsl(252 100% 68%)" />
            <stop offset="100%" stopColor="hsl(339 100% 65%)" />
          </linearGradient>
        </defs>

        {/* Track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="hsl(0 0% 100% / 0.06)"
          strokeWidth={stroke}
        />
        {/* Progress */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="url(#ring-pv)"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={offset}
          style={{
            transition: "stroke-dashoffset 600ms cubic-bezier(0.22,1,0.36,1)",
            filter: "drop-shadow(0 0 6px hsl(339 100% 65% / 0.55))",
          }}
        />
      </svg>

      {/* Center text */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-[28px] font-light leading-none tracking-tight text-foreground">
          {Math.round(value)}
          <span className="text-[16px] text-muted-foreground">%</span>
        </span>
        <span className="mt-1 text-[10px] lowercase tracking-[0.18em] text-muted-foreground">
          confidence
        </span>
      </div>
    </div>
  );
};
