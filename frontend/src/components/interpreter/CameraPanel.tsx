import { useEffect, useRef, useState } from "react";

/**
 * Stylized hand visualization — fingertip glow dots, soft halo,
 * and a smooth trailing path following the index fingertip
 * (gradient pink → violet, fading out).
 */
const HandVisual = () => {
  const tips = [
    { x: 38, y: 38 }, // thumb
    { x: 47, y: 22 }, // index
    { x: 55, y: 17 }, // middle
    { x: 63, y: 22 }, // ring
    { x: 71, y: 30 }, // pinky
  ];
  const indexTip = tips[1];

  // Animated trail points behind index fingertip
  const [tick, setTick] = useState(0);
  const raf = useRef<number>();
  useEffect(() => {
    const start = performance.now();
    const loop = (t: number) => {
      setTick((t - start) / 1000);
      raf.current = requestAnimationFrame(loop);
    };
    raf.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf.current!);
  }, []);

  // 12 trail dots forming a smooth, slightly delayed path
  const trail = Array.from({ length: 14 }).map((_, i) => {
    const phase = tick * 1.2 - i * 0.07;
    const wave = Math.sin(phase) * 6;
    const x = indexTip.x + Math.cos(phase * 0.9) * 4;
    const y = indexTip.y - i * 1.4 + wave * 0.4 - 2;
    return { x, y, opacity: Math.max(0, 1 - i / 14) };
  });

  return (
    <div className="absolute inset-0">
      {/* Halo */}
      <div
        aria-hidden
        className="absolute left-1/2 top-1/2 h-[55%] w-[55%] -translate-x-1/2 -translate-y-[48%] rounded-full"
        style={{
          background:
            "radial-gradient(circle, hsl(339 100% 65% / 0.15), hsl(252 100% 68% / 0.08) 45%, transparent 70%)",
          filter: "blur(10px)",
        }}
      />

      {/* Faint palm */}
      <div
        aria-hidden
        className="absolute left-1/2 top-[60%] h-[28%] w-[26%] -translate-x-1/2 rounded-[40%] opacity-25"
        style={{
          background:
            "radial-gradient(ellipse, hsl(0 0% 100% / 0.18), transparent 65%)",
          filter: "blur(8px)",
        }}
      />

      {/* Connection lines (very faint) */}
      <svg className="absolute inset-0 h-full w-full opacity-25" viewBox="0 0 100 100" preserveAspectRatio="none">
        <g stroke="hsl(0 0% 100%)" strokeOpacity="0.4" strokeWidth="0.2" fill="none" strokeLinecap="round">
          {tips.map((t, i) => (
            <line key={i} x1="55" y1="60" x2={t.x} y2={t.y} />
          ))}
        </g>
      </svg>

      {/* Fingertip glow dots */}
      {tips.map((t, i) => (
        <div
          key={i}
          className="absolute -translate-x-1/2 -translate-y-1/2"
          style={{ left: `${t.x}%`, top: `${t.y}%` }}
        >
          <div
            className="h-2 w-2 rounded-full bg-white animate-dot-pulse"
            style={{
              animationDelay: `${i * 120}ms`,
              boxShadow: "0 0 10px hsl(339 100% 65% / 0.9), 0 0 22px hsl(252 100% 68% / 0.5)",
            }}
          />
        </div>
      ))}

      {/* Trail following index fingertip */}
      <svg className="absolute inset-0 h-full w-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
        <defs>
          <linearGradient id="trail-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="hsl(339 100% 65%)" stopOpacity="0.95" />
            <stop offset="1" stopColor="hsl(252 100% 68%)" stopOpacity="0" />
          </linearGradient>
        </defs>
        <polyline
          points={trail.map((p) => `${p.x},${p.y}`).join(" ")}
          stroke="url(#trail-grad)"
          strokeWidth="1.2"
          strokeLinecap="round"
          fill="none"
          style={{ filter: "drop-shadow(0 0 4px hsl(339 100% 65% / 0.6))" }}
        />
        {trail.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r={0.8 - i * 0.04} fill="hsl(339 100% 65%)" opacity={p.opacity * 0.7} />
        ))}
      </svg>
    </div>
  );
};

export const CameraPanel = () => {
  return (
    <div
      className="camera-inner-glow relative aspect-video w-full overflow-hidden rounded-3xl bg-black"
      style={{ borderRadius: 24 }}
    >
      {/* Subtle dark gradient inside */}
      <div
        aria-hidden
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at center, hsl(0 0% 6%) 0%, hsl(0 0% 2%) 70%, hsl(0 0% 0%) 100%)",
        }}
      />

      {/* Vignette */}
      <div
        aria-hidden
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at center, transparent 55%, hsl(0 0% 0% / 0.7) 100%)",
        }}
      />

      <HandVisual />

      {/* Top-left REC */}
      <div className="absolute left-5 top-5 flex items-center gap-2">
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-primary glow-pink-soft" />
        </span>
        <span className="font-mono text-[10.5px] uppercase tracking-[0.22em] text-foreground/80">
          Live
        </span>
      </div>

      {/* Top-right resolution */}
      <div className="absolute right-5 top-5 font-mono text-[10.5px] uppercase tracking-[0.18em] text-muted-foreground">
        1080p · 60fps
      </div>

      {/* Corner brackets */}
      {[
        "left-4 top-4 border-l border-t",
        "right-4 top-4 border-r border-t",
        "left-4 bottom-4 border-l border-b",
        "right-4 bottom-4 border-r border-b",
      ].map((c) => (
        <span
          key={c}
          className={`pointer-events-none absolute h-3 w-3 rounded-[2px] border-white/15 ${c}`}
        />
      ))}
    </div>
  );
};
