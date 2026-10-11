import clsx from "clsx";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { MODALITY_COLOR, MODALITY_LABEL } from "../lib/palette";

interface Stage {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  lines: string[];
  to: string;
  info: string;
}

const STAGES: Stage[] = [
  { id: "prep", x: 178, y: 92, w: 128, h: 120, title: "Preprocessing", lines: ["align · resample", "window · normalise"], to: "/data", info: "Synchronized wrist streams are resampled to model rates, cut into label-pure windows and z-normalised with training-fold statistics only." },
  { id: "enc", x: 346, y: 92, w: 132, h: 120, title: "Modality encoders", lines: ["compact 1-D CNN", "per stream → h_m"], to: "/attack", info: "Each modality has its own temporal encoder, so modality-specific contributions stay measurable. Auxiliary unimodal heads expose per-modality opinions." },
  { id: "rel", x: 518, y: 92, w: 136, h: 120, title: "Reliability engine", lines: ["Q  signal quality", "D  disagreement", "U  uncertainty"], to: "/reliability", info: "Evidence is ranked against clean validation data into a conformal reliability R_m (frozen before any attack is evaluated)." },
  { id: "fuse", x: 694, y: 92, w: 136, h: 120, title: "Attack-aware fusion", lines: ["w_m from R_m", "gate h_m → μ_m", "low-confidence flag"], to: "/attack", info: "The same frozen classifier receives reliability-gated embeddings. Unreliable modalities lose influence, and a flag fires below α/M." },
  { id: "out", x: 870, y: 92, w: 120, h: 120, title: "Output", lines: ["prediction", "reliability R", "attack detected?"], to: "/results", info: "Every decision is logged with its attack parameters and reliability scores (traceable manifests)." },
];

export default function PipelineDiagram({ modalities }: { modalities: string[] }) {
  const [hover, setHover] = useState<string | null>(null);
  const navigate = useNavigate();
  const info = STAGES.find((s) => s.id === hover)?.info ?? (hover === "attack" ? "SMSA: mask, attenuate or burst-drop exactly one modality at inference time; all other streams pass through unchanged." : hover === "sensors" ? "Wrist-worn Empatica E4: BVP 64 Hz, ACC 32 Hz, EDA and skin temperature 4 Hz." : "Hover a stage for details; click to open its page.");
  const sensorY = (i: number) => 60 + i * (180 / Math.max(modalities.length, 1));
  return (
    <div>
      <svg viewBox="0 0 1000 250" className="w-full" role="img" aria-label="TrustFuse system architecture">
        <defs>
          <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="#98a2b3" />
          </marker>
          <marker id="arrA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="#d03b3b" />
          </marker>
        </defs>
        {/* sensors */}
        <g onMouseEnter={() => setHover("sensors")} onMouseLeave={() => setHover(null)} className="cursor-pointer" onClick={() => navigate("/data")}>
          <rect x={10} y={40} width={128} height={196} rx={12} fill="#f9fafb" stroke={hover === "sensors" ? "#101828" : "#e4e7ec"} strokeDasharray="4 3" />
          <text x={74} y={30} textAnchor="middle" className="fill-muted text-[11px] font-medium uppercase tracking-wider">wearable sensors</text>
          {modalities.map((m, i) => (
            <g key={m} transform={`translate(22, ${sensorY(i) - 14})`}>
              <rect width={104} height={28} rx={7} fill="#fff" stroke="#e4e7ec" />
              <circle cx={14} cy={14} r={4.5} fill={MODALITY_COLOR[m]} />
              <text x={26} y={18} className="fill-ink text-[12px] font-medium">{MODALITY_LABEL[m]}</text>
              <path d={`M 66 14 q 4 -8 8 0 t 8 0 t 8 0`} fill="none" stroke={MODALITY_COLOR[m]} strokeWidth={1.4} />
            </g>
          ))}
        </g>
        {modalities.map((m, i) => (
          <path key={m} d={`M 126 ${sensorY(i)} C 150 ${sensorY(i)}, 155 152, 176 152`} fill="none" stroke="#c4c9d2" strokeWidth={1.2} />
        ))}
        {/* attack surface */}
        <g onMouseEnter={() => setHover("attack")} onMouseLeave={() => setHover(null)} className="cursor-pointer" onClick={() => navigate("/attack")}>
          <rect x={330} y={4} width={164} height={52} rx={10} fill="#fdecec" stroke={hover === "attack" ? "#d03b3b" : "#f3b8b8"} />
          <text x={412} y={24} textAnchor="middle" className="text-[11.5px] font-semibold" fill="#b42318">ATTACK SURFACE · SMSA</text>
          <text x={412} y={42} textAnchor="middle" className="text-[11px]" fill="#b42318">mask · attenuate · burst · noise</text>
        </g>
        <path d="M 412 56 L 412 88" stroke="#d03b3b" strokeWidth={1.6} markerEnd="url(#arrA)" strokeDasharray="5 4" className="animate-flow" />
        {/* stages */}
        {STAGES.map((s, i) => (
          <g key={s.id} onMouseEnter={() => setHover(s.id)} onMouseLeave={() => setHover(null)} onClick={() => navigate(s.to)} className="cursor-pointer">
            <rect x={s.x} y={s.y} width={s.w} height={s.h} rx={12} fill={s.id === "rel" || s.id === "fuse" ? "#f5f9fe" : "#fff"} stroke={hover === s.id ? "#101828" : s.id === "rel" || s.id === "fuse" ? "#b9d3f3" : "#e4e7ec"} strokeWidth={hover === s.id ? 1.5 : 1} />
            <text x={s.x + s.w / 2} y={s.y + 28} textAnchor="middle" className="fill-ink text-[12.5px] font-semibold">{s.title}</text>
            {s.lines.map((l, k) => (
              <text key={l} x={s.x + s.w / 2} y={s.y + 54 + k * 19} textAnchor="middle" className={clsx("text-[11px]", "fill-muted")}>{l}</text>
            ))}
            {i < STAGES.length - 1 && (
              <path d={`M ${s.x + s.w + 3} ${s.y + s.h / 2} L ${STAGES[i + 1].x - 4} ${s.y + s.h / 2}`} stroke="#98a2b3" strokeWidth={1.4} markerEnd="url(#arr)" strokeDasharray="6 4" className="animate-flow" />
            )}
          </g>
        ))}
        <path d="M 306 152 L 342 152" stroke="#98a2b3" strokeWidth={1.4} markerEnd="url(#arr)" strokeDasharray="6 4" className="animate-flow" />
        <text x={762} y={236} textAnchor="middle" className="fill-faint text-[10.5px]">same frozen checkpoint for fixed and defended pipelines</text>
      </svg>
      <div className="mt-2 min-h-[38px] rounded-lg bg-sunken px-3 py-2 text-[12.5px] text-ink-2 ring-1 ring-inset ring-line">{info}</div>
    </div>
  );
}
