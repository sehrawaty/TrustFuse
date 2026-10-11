import clsx from "clsx";
import { motion } from "framer-motion";
import { CheckCircle2, XCircle } from "lucide-react";
import type { Head, ModalityEvidence } from "../lib/api";
import { MODALITY_COLOR, MODALITY_LABEL } from "../lib/palette";
import { Badge, InfoTip, Tip } from "./ui";

/** Horizontal class-probability bars for one fusion pipeline. */
export function ProbBars({ head, classes, tone, truth, cleanPred }: { head: Head; classes: string[]; tone: "fixed" | "defended"; truth: number; cleanPred?: number }) {
  return (
    <div className="space-y-1.5">
      {classes.map((c, i) => {
        const p = head.probs[i];
        const isPred = head.pred === i;
        return (
          <div key={c} className="grid grid-cols-[minmax(84px,auto)_1fr_48px] items-center gap-2.5">
            <div className={clsx("flex items-center gap-1 truncate text-[12.5px]", isPred ? "font-medium text-ink" : "text-muted")}>
              {c}
              {i === truth && (
                <Tip content="Ground-truth label of this window">
                  <span className="rounded bg-sunken px-1 text-[10px] font-medium text-muted ring-1 ring-line">true</span>
                </Tip>
              )}
            </div>
            <div className="relative h-2.5 overflow-hidden rounded-full bg-sunken ring-1 ring-inset ring-line">
              <motion.div
                className={clsx("absolute inset-y-0 left-0 rounded-full", tone === "fixed" ? "bg-fixed" : "bg-defended", !isPred && "opacity-40")}
                initial={false}
                animate={{ width: `${Math.max(p * 100, 0.5)}%` }}
                transition={{ type: "spring", stiffness: 180, damping: 26 }}
              />
              {cleanPred === i && <div className="absolute inset-y-0 w-px bg-ink/40" style={{ left: "calc(100% - 1px)" }} />}
            </div>
            <div className={clsx("num text-right text-[12px]", isPred ? "text-ink" : "text-muted")}>{(p * 100).toFixed(1)}%</div>
          </div>
        );
      })}
    </div>
  );
}

export function Verdict({ head, classes, truth, changed }: { head: Head; classes: string[]; truth: number; changed: boolean }) {
  const ok = head.pred === truth;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-[17px] font-semibold text-ink">{classes[head.pred]}</span>
      {ok ? (
        <Badge tone="good" icon={<CheckCircle2 className="h-3 w-3" />}>
          correct
        </Badge>
      ) : (
        <Badge tone="attack" icon={<XCircle className="h-3 w-3" />}>
          wrong
        </Badge>
      )}
      {changed && <Badge tone="dark">decision flipped by attack</Badge>}
    </div>
  );
}

/** Reliability of one modality: R on a log scale with the flag threshold, plus the gate weight. */
export function ReliabilityRow({ m, ev, threshold, target, components }: { m: string; ev: ModalityEvidence; threshold: number; target: boolean; components: string[] }) {
  // log10 scale from 1e-3 to 1
  const pos = (r: number) => Math.min(100, Math.max(0, ((Math.log10(Math.max(r, 1e-3)) + 3) / 3) * 100));
  const flagged = ev.R <= threshold;
  const parts = [
    { k: "Q", v: ev.Q, label: "signal quality anomaly" },
    { k: "D", v: ev.zD, label: "disagreement z" },
    { k: "U", v: ev.zU, label: "uncertainty z" },
  ];
  return (
    <div className={clsx("rounded-lg border px-3 py-2.5 transition-colors", target ? "border-attack/30 bg-attack-soft/40" : "border-line bg-surface")}>
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: MODALITY_COLOR[m] }} />
          <span className="text-[13px] font-semibold">{MODALITY_LABEL[m]}</span>
          {target && <span className="text-[11px] font-medium text-attack">attacked</span>}
        </div>
        <div className="flex items-center gap-3 text-[12px]">
          <span className="text-muted">
            R <span className={clsx("num font-medium", flagged ? "text-flag-ink" : "text-ink")}>{ev.R < 0.001 ? "<0.001" : ev.R.toFixed(3)}</span>
          </span>
          <span className="text-muted">
            w <span className="num font-medium text-ink">{ev.w.toFixed(2)}</span>
          </span>
        </div>
      </div>
      <Tip content={`Conformal reliability R = ${ev.R.toFixed(4)} (log scale). The flag fires when any modality falls at or below α/M = ${threshold.toFixed(4)}.`}>
        <div className="relative h-2 rounded-full bg-gradient-to-r from-[#fde4e1] via-[#fff3d6] to-[#e7f6ef]">
          <div className="absolute inset-y-[-3px] w-px bg-ink/50" style={{ left: `${pos(threshold)}%` }} />
          <motion.div
            className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow-card"
            style={{ background: MODALITY_COLOR[m] }}
            initial={false}
            animate={{ left: `${pos(ev.R)}%` }}
            transition={{ type: "spring", stiffness: 160, damping: 22 }}
          />
        </div>
      </Tip>
      <div className="mt-2.5 grid grid-cols-[1fr_auto] items-center gap-3">
        <div className="flex gap-1.5">
          {parts.map((p) => (
            <Tip key={p.k} content={`${p.k}: ${p.label} = ${p.v.toFixed(2)}${components.includes(p.k) ? "" : " (disabled in this ablation)"}`}>
              <span
                className={clsx(
                  "num inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] ring-1 ring-inset",
                  !components.includes(p.k) ? "text-faint ring-line line-through" : p.v > 3 ? "bg-flag-soft text-flag-ink ring-flag/40" : "text-ink-2 ring-line",
                )}
              >
                {p.k} {p.v.toFixed(1)}
              </span>
            </Tip>
          ))}
        </div>
        <Tip content="Fusion gate weight w: 1 keeps the modality embedding; 0 replaces it with the clean-validation reference.">
          <div className="flex w-24 items-center gap-1.5">
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-line">
              <motion.div className="h-full rounded-full bg-defended" initial={false} animate={{ width: `${ev.w * 100}%` }} />
            </div>
          </div>
        </Tip>
      </div>
    </div>
  );
}

export function EvidenceLegend() {
  return (
    <div className="flex items-center gap-1 text-xs text-muted">
      Q · D · U → R → w
      <InfoTip>
        Q: model-free signal-quality anomaly. D: disagreement of the modality's unimodal head with the other heads (z-score). U: MC-dropout predictive
        uncertainty (z-score). Their sum is ranked against clean validation data to give the conformal reliability R; the fusion weight w closes once the
        anomaly passes the calibrated threshold.
      </InfoTip>
    </div>
  );
}
