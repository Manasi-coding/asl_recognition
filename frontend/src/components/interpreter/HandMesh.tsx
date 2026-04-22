import { useMemo } from "react";

/**
 * Stylized wireframe hand visual for a given letter.
 * - Mesh/skeleton lines (faint)
 * - Glowing pink landmark points
 * - Soft pink/violet glow halo
 * - Optional motion trail for dynamic letters (J, Z)
 *
 * Shapes are intentionally abstract — not anatomically correct.
 * Each letter has a deterministic pose so the grid feels varied.
 */

type Pose = {
  // Landmark points in a 100x100 viewBox. First = wrist, then chains per finger.
  points: Array<[number, number]>;
  // Connections as index pairs into `points`.
  edges: Array<[number, number]>;
};

// Base pose: wrist + 5 finger chains of 3 joints (tip last)
// indices: 0 wrist
//   thumb: 1,2,3
//   index: 4,5,6
//   middle: 7,8,9
//   ring: 10,11,12
//   pinky: 13,14,15
const makePose = (
  fingers: Array<{ base: [number, number]; mid: [number, number]; tip: [number, number] }>,
  wrist: [number, number] = [50, 82]
): Pose => {
  const points: Array<[number, number]> = [wrist];
  const edges: Array<[number, number]> = [];
  fingers.forEach((f, i) => {
    const baseIdx = points.length;
    points.push(f.base, f.mid, f.tip);
    // wrist → base → mid → tip
    edges.push([0, baseIdx], [baseIdx, baseIdx + 1], [baseIdx + 1, baseIdx + 2]);
    // cross-connect bases (palm mesh)
    if (i > 0) edges.push([baseIdx - 3, baseIdx]);
  });
  return { points, edges };
};

const POSES: Record<string, Pose> = {
  // Open palm — default for most letters
  open: makePose([
    { base: [30, 62], mid: [24, 52], tip: [20, 44] }, // thumb
    { base: [38, 58], mid: [36, 38], tip: [35, 22] }, // index
    { base: [48, 58], mid: [48, 34], tip: [48, 16] }, // middle
    { base: [58, 58], mid: [60, 36], tip: [62, 20] }, // ring
    { base: [68, 62], mid: [72, 48], tip: [76, 34] }, // pinky
  ]),
  // Closed fist — A, E, M, N, S, T
  fist: makePose([
    { base: [32, 62], mid: [28, 54], tip: [34, 50] },
    { base: [40, 58], mid: [40, 48], tip: [44, 52] },
    { base: [48, 58], mid: [48, 46], tip: [50, 52] },
    { base: [56, 58], mid: [58, 48], tip: [56, 52] },
    { base: [64, 62], mid: [66, 52], tip: [62, 54] },
  ]),
  // Index pointing up — D, I, J, Z, 1
  indexUp: makePose([
    { base: [32, 62], mid: [30, 54], tip: [36, 50] },
    { base: [40, 58], mid: [40, 38], tip: [40, 18] }, // index extended
    { base: [48, 58], mid: [50, 50], tip: [48, 56] },
    { base: [56, 58], mid: [58, 50], tip: [54, 56] },
    { base: [64, 62], mid: [66, 54], tip: [62, 58] },
  ]),
  // Two fingers up — U, V, K, R
  twoUp: makePose([
    { base: [32, 62], mid: [30, 54], tip: [34, 50] },
    { base: [40, 58], mid: [40, 38], tip: [40, 20] },
    { base: [50, 58], mid: [52, 38], tip: [54, 20] },
    { base: [58, 58], mid: [60, 50], tip: [56, 56] },
    { base: [66, 62], mid: [68, 54], tip: [64, 58] },
  ]),
  // Pinky out — I, Y
  pinkyOut: makePose([
    { base: [32, 62], mid: [28, 54], tip: [34, 50] },
    { base: [40, 58], mid: [40, 48], tip: [44, 52] },
    { base: [48, 58], mid: [48, 46], tip: [50, 52] },
    { base: [56, 58], mid: [58, 48], tip: [56, 52] },
    { base: [66, 62], mid: [70, 48], tip: [74, 32] },
  ]),
  // Thumb + pinky (Y) / hang-loose
  thumbPinky: makePose([
    { base: [30, 62], mid: [22, 52], tip: [16, 42] },
    { base: [40, 58], mid: [40, 50], tip: [44, 54] },
    { base: [48, 58], mid: [48, 50], tip: [50, 54] },
    { base: [56, 58], mid: [58, 50], tip: [54, 54] },
    { base: [66, 62], mid: [72, 52], tip: [78, 40] },
  ]),
  // OK-circle — O, F
  ok: makePose([
    { base: [30, 58], mid: [28, 48], tip: [36, 40] },
    { base: [38, 54], mid: [34, 42], tip: [40, 36] }, // index curls to meet thumb
    { base: [48, 54], mid: [50, 36], tip: [52, 20] },
    { base: [56, 54], mid: [60, 36], tip: [62, 22] },
    { base: [64, 58], mid: [70, 44], tip: [74, 30] },
  ]),
  // L shape — L, G
  lShape: makePose([
    { base: [30, 62], mid: [22, 56], tip: [14, 52] }, // thumb out
    { base: [40, 58], mid: [40, 38], tip: [40, 20] }, // index up
    { base: [48, 58], mid: [50, 50], tip: [48, 56] },
    { base: [56, 58], mid: [58, 50], tip: [54, 56] },
    { base: [64, 62], mid: [66, 54], tip: [62, 58] },
  ]),
  // C — curved
  cShape: makePose([
    { base: [34, 52], mid: [28, 40], tip: [30, 28] },
    { base: [40, 50], mid: [36, 32], tip: [42, 20] },
    { base: [48, 50], mid: [48, 28], tip: [56, 18] },
    { base: [56, 52], mid: [60, 32], tip: [66, 24] },
    { base: [62, 56], mid: [68, 40], tip: [70, 32] },
  ]),
  // Three up — W, 3
  threeUp: makePose([
    { base: [30, 62], mid: [24, 54], tip: [28, 50] },
    { base: [40, 58], mid: [40, 38], tip: [40, 20] },
    { base: [50, 58], mid: [50, 34], tip: [50, 16] },
    { base: [60, 58], mid: [62, 38], tip: [64, 20] },
    { base: [66, 62], mid: [70, 54], tip: [66, 58] },
  ]),
  // Thumb up / out — A, T
  thumbUp: makePose([
    { base: [28, 58], mid: [22, 46], tip: [18, 32] },
    { base: [40, 58], mid: [40, 48], tip: [44, 52] },
    { base: [48, 58], mid: [48, 46], tip: [50, 52] },
    { base: [56, 58], mid: [58, 48], tip: [56, 52] },
    { base: [64, 62], mid: [66, 52], tip: [62, 54] },
  ]),
};

// Map each letter to a pose key.
const LETTER_POSE: Record<string, keyof typeof POSES> = {
  A: "fist", B: "open", C: "cShape", D: "indexUp", E: "fist",
  F: "ok", G: "lShape", H: "twoUp", I: "pinkyOut", J: "indexUp",
  K: "twoUp", L: "lShape", M: "fist", N: "fist", O: "ok",
  P: "twoUp", Q: "lShape", R: "twoUp", S: "fist", T: "thumbUp",
  U: "twoUp", V: "twoUp", W: "threeUp", X: "indexUp", Y: "thumbPinky",
  Z: "indexUp",
};

interface HandMeshProps {
  letter: string;
  showJPath?: boolean;
  showZPath?: boolean;
}

export const HandMesh = ({ letter, showJPath, showZPath }: HandMeshProps) => {
  const pose = POSES[LETTER_POSE[letter] ?? "open"];

  const { points, edges } = pose;

  // Stable fingertip indices (tip = base+2 = indices 3, 6, 9, 12, 15)
  const tipIdx = useMemo(() => [3, 6, 9, 12, 15], []);
  const indexTip = points[6];

  return (
    <div className="relative h-full w-full">
      {/* Soft halo behind the hand */}
      <div
        aria-hidden
        className="absolute left-1/2 top-[58%] h-[70%] w-[70%] -translate-x-1/2 -translate-y-1/2 rounded-full"
        style={{
          background:
            "radial-gradient(circle, hsl(339 100% 65% / 0.22), hsl(252 100% 68% / 0.10) 45%, transparent 72%)",
          filter: "blur(10px)",
        }}
      />

      {/* Mesh lines + motion paths */}
      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="xMidYMid meet"
        className="absolute inset-0 h-full w-full"
      >
        <defs>
          <linearGradient id={`mesh-grad-${letter}`} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="hsl(0 0% 100%)" stopOpacity="0.55" />
            <stop offset="1" stopColor="hsl(339 100% 65%)" stopOpacity="0.55" />
          </linearGradient>
          <linearGradient id={`trail-grad-${letter}`} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="hsl(339 100% 65%)" />
            <stop offset="1" stopColor="hsl(252 100% 68%)" />
          </linearGradient>
          <marker
            id={`trail-arrow-${letter}`}
            markerWidth="5"
            markerHeight="5"
            refX="2.5"
            refY="2.5"
            orient="auto"
          >
            <path d="M0,0 L5,2.5 L0,5 Z" fill="hsl(252 100% 68%)" />
          </marker>
        </defs>

        {/* Skeleton edges */}
        <g
          stroke={`url(#mesh-grad-${letter})`}
          strokeWidth="0.6"
          strokeLinecap="round"
          fill="none"
          opacity="0.75"
        >
          {edges.map(([a, b], i) => (
            <line
              key={i}
              x1={points[a][0]}
              y1={points[a][1]}
              x2={points[b][0]}
              y2={points[b][1]}
            />
          ))}
          {/* Palm outline triangle for depth */}
          <path
            d={`M${points[1][0]},${points[1][1]} L${points[13][0]},${points[13][1]} L${points[0][0]},${points[0][1]} Z`}
            opacity="0.35"
          />
        </g>

        {/* Motion trail for J (curved hook down from index fingertip) */}
        {showJPath && (
          <path
            d={`M${indexTip[0]},${indexTip[1]} C${indexTip[0] + 2},${indexTip[1] + 18} ${indexTip[0] - 8},${indexTip[1] + 28} ${indexTip[0] - 18},${indexTip[1] + 22}`}
            stroke={`url(#trail-grad-${letter})`}
            strokeWidth="1.4"
            strokeLinecap="round"
            fill="none"
            markerEnd={`url(#trail-arrow-${letter})`}
            style={{ filter: "drop-shadow(0 0 4px hsl(339 100% 65% / 0.7))" }}
          />
        )}

        {/* Motion trail for Z (→ ↙ →) */}
        {showZPath && (
          <path
            d={`M${indexTip[0] - 10},${indexTip[1] - 4} L${indexTip[0] + 14},${indexTip[1] - 4} L${indexTip[0] - 10},${indexTip[1] + 14} L${indexTip[0] + 14},${indexTip[1] + 14}`}
            stroke={`url(#trail-grad-${letter})`}
            strokeWidth="1.4"
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
            markerEnd={`url(#trail-arrow-${letter})`}
            style={{ filter: "drop-shadow(0 0 4px hsl(339 100% 65% / 0.7))" }}
          />
        )}

        {/* Landmark points — all joints faint, tips brighter */}
        {points.map(([x, y], i) => {
          const isTip = tipIdx.includes(i);
          return (
            <circle
              key={i}
              cx={x}
              cy={y}
              r={isTip ? 1.4 : 0.9}
              fill={isTip ? "hsl(339 100% 65%)" : "hsl(0 0% 100%)"}
              opacity={isTip ? 1 : 0.75}
              style={
                isTip
                  ? { filter: "drop-shadow(0 0 3px hsl(339 100% 65% / 0.9))" }
                  : undefined
              }
            />
          );
        })}
      </svg>
    </div>
  );
};
