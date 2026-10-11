import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { ChevronLeft, ChevronRight, Crosshair, RotateCcw, ScanSearch, ShieldAlert, Sparkles } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import EChart, { ttRow, ttTitle } from "../components/charts/EChart";
import SignalLane from "../components/charts/SignalLane";
import { EvidenceLegend, ProbBars, ReliabilityRow, Verdict } from "../components/Evidence";
import SequenceTrace from "../components/SequenceTrace";
import {
  AttackBadge, Badge, Button, Card, CardHeader, Empty, ErrorNote, FlagBadge, InfoTip, Kbd, PageHeader, Segmented, Select, Skeleton, Slider,
} from "../components/ui";
import WindowStrip from "../components/WindowStrip";
import { api, type AdaptiveResult, type Attack, type Fill, type InferResult, type TargetedResult } from "../lib/api";
import { clock } from "../lib/format";
import { COLORS, MODALITY_COLOR, MODALITY_LABEL, MODALITY_LONG, MODALITY_SHORT } from "../lib/palette";
import { DEFAULT_ATTACK, defaultWindow, useApp, windowKey } from "../lib/store";

function useDebounced<T>(value: T, ms = 160): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(id);
  }, [value, ms]);
  return v;
}

const TYPE_HELP: Record<string, string> = {
  none: "No attack: both pipelines see the clean window.",
  mask: "Replace a contiguous share of the window. Zero: the sensor reports zeros. Missing: samples are dropped (NaN, imputed by the model). Hold: the sensor freezes at its last value.",
  attenuate: "Scale the whole target signal by (1 − severity), i.e. partial suppression such as a loosened strap.",
  noise: "Additive Gaussian corruption at a chosen signal-to-noise ratio (extension).",
};

export default function AttackLab() {
  const { task, subject: subjMap, setSubject, window: winMap, setWindow, attack, setAttack, variant, setVariant, components, toggleComponent, calibration, setCalibration } = useApp();
  const overview = useQuery({ queryKey: ["overview"], queryFn: api.overview });
  const taskInfo = overview.data?.tasks.find((t) => t.key === task);
  const subjects = useQuery({ queryKey: ["subjects", task], queryFn: () => api.subjects(task) });
  const subject = subjMap[task] ?? subjects.data?.[0]?.subject ?? "S2";
  const windows = useQuery({ queryKey: ["windows", task, subject], queryFn: () => api.windows(task, subject) });
  const key = windowKey(task, subject);
  const nWin = windows.data?.start_s.length ?? 0;
  const win = Math.min(winMap[key] ?? defaultWindow(windows.data), Math.max(nWin - 1, 0));
  const mods = taskInfo?.modalities ?? ["bvp", "acc", "eda", "temp"];

  const debounced = useDebounced({ attack, win, components: components.join(""), variant, calibration });
  const infer = useQuery({
    queryKey: ["infer", task, subject, debounced],
    queryFn: () =>
      api.infer({ task, subject, window: debounced.win, variant: debounced.variant, attack: debounced.attack, components: debounced.components.split(""), calibration: debounced.calibration }),
    enabled: !!windows.data && nWin > 0,
    placeholderData: keepPreviousData,
  });
  const [version, setVersion] = useState(0);
  useEffect(() => setVersion((v) => v + 1), [infer.data]);

  const targeted = useMutation({ mutationFn: () => api.targeted({ task, subject, window: win, fill: attack.fill, variant }) });
  const adaptive = useMutation({ mutationFn: () => api.adaptive({ task, subject, window: win, variant }) });
  useEffect(() => {
    targeted.reset();
    adaptive.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task, subject, win, variant]);

  // keyboard: 1-4 target modality, 0 clean, [ ] previous / next window
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest("input,textarea,[role=slider]")) return;
      const n = Number(e.key);
      if (n >= 1 && n <= mods.length) setAttack({ modality: mods[n - 1], type: attack.type === "none" ? "mask" : attack.type });
      if (e.key === "0") setAttack({ type: "none" });
      if (e.key === "]") setWindow(key, Math.min(nWin - 1, win + 1));
      if (e.key === "[") setWindow(key, Math.max(0, win - 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mods, attack.type, setAttack, setWindow, key, nWin, win]);

  const r = infer.data;
  const attackOn = attack.type !== "none";

  return (
    <div>
      <PageHeader
        eyebrow="Attack & defend"
        title="Attack Lab"
        description={
          <>
            Suppress one sensor stream of a real held-out window and compare the two pipelines side by side. Both receive the <em>identical</em> attacked
            window and share one frozen checkpoint. Only the fusion step differs.
          </>
        }
        actions={
          <Button variant="ghost" icon={<RotateCcw className="h-3.5 w-3.5" />} onClick={() => setAttack({ ...DEFAULT_ATTACK, modality: mods.includes("eda") ? "eda" : mods[0] })}>
            Reset attack
          </Button>
        }
      />

      <Card className="mb-4">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3 px-5 py-3.5">
          <div className="flex items-center gap-2">
            <span className="eyebrow">Subject</span>
            <Select
              label="Subject"
              value={subject}
              onChange={(s) => setSubject(task, s)}
              options={(subjects.data ?? []).map((s) => ({ value: s.subject, label: s.subject, hint: `${s.windows} windows` }))}
              className="w-[150px]"
            />
            <InfoTip>Every subject is served by the LOSO fold model that held that subject out, so every prediction here is on unseen data.</InfoTip>
          </div>
          <div className="flex items-center gap-2">
            <span className="eyebrow">Window</span>
            <Button size="sm" variant="secondary" aria-label="Previous window" onClick={() => setWindow(key, Math.max(0, win - 1))}>
              <ChevronLeft className="h-3.5 w-3.5" />
            </Button>
            <span className="num w-[170px] whitespace-nowrap text-center text-[12.5px]">
              #{win + 1} / {nWin} · {windows.data ? clock(windows.data.start_s[win] ?? 0) : "—"}
            </span>
            <Button size="sm" variant="secondary" aria-label="Next window" onClick={() => setWindow(key, Math.min(nWin - 1, win + 1))}>
              <ChevronRight className="h-3.5 w-3.5" />
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <span className="eyebrow">Model</span>
            <Segmented
              size="sm"
              value={variant}
              onChange={setVariant}
              options={[
                { value: "fusion", label: "Standard fusion", title: "Clean-trained concatenation fusion (the report's baseline)" },
                { value: "moddrop", label: "ModDrop", title: "Same architecture trained with random modality dropout (robustness baseline)" },
              ]}
            />
          </div>
          <div className="ml-auto flex items-center gap-1.5 text-[11.5px] text-muted">
            <Kbd>1</Kbd>–<Kbd>{mods.length}</Kbd> target <Kbd>0</Kbd> clean <Kbd>[</Kbd>
            <Kbd>]</Kbd> window
          </div>
        </div>
        <div className="border-t border-line px-5 pb-3 pt-3">
          {windows.data ? (
            <WindowStrip
              start={windows.data.start_s}
              y={windows.data.y}
              classes={windows.data.classes}
              value={win}
              onChange={(i) => setWindow(key, i)}
              windowS={windows.data.window_s}
            />
          ) : (
            <Skeleton className="h-12" />
          )}
        </div>
      </Card>

      <div className="grid grid-cols-12 gap-4">
        {/* ---------------- attack configuration ---------------- */}
        <Card className="col-span-12 xl:col-span-3">
          <CardHeader eyebrow="Threat" title="Selective Modality Suppression" sub="One target stream is modified; all others pass through untouched." />
          <div className="space-y-5 p-5">
            <div>
              <div className="eyebrow mb-2">Target modality</div>
              <div className="grid grid-cols-2 gap-2">
                {mods.map((m, i) => (
                  <button
                    key={m}
                    onClick={() => setAttack({ modality: m, type: attack.type === "none" ? "mask" : attack.type })}
                    className={clsx(
                      "flex items-center gap-2 rounded-lg border px-2.5 py-2 text-left transition-all",
                      attack.modality === m && attackOn ? "border-ink bg-surface shadow-card ring-1 ring-ink" : "border-line hover:border-line-strong",
                    )}
                  >
                    <span className="h-2.5 w-2.5 rounded-full" style={{ background: MODALITY_COLOR[m] }} />
                    <span className="min-w-0">
                      <span className="block text-[13px] font-medium">{MODALITY_LABEL[m]}</span>
                      <span className="block truncate text-[10.5px] text-muted" title={MODALITY_LONG[m]}>{MODALITY_SHORT[m]}</span>
                    </span>
                    <Kbd>{i + 1}</Kbd>
                  </button>
                ))}
              </div>
            </div>
            <div>
              <div className="eyebrow mb-2 flex items-center gap-1">
                Attack <InfoTip>{TYPE_HELP[attack.type]}</InfoTip>
              </div>
              <Segmented
                value={attack.type}
                onChange={(t) => setAttack({ type: t as Attack["type"] })}
                options={[
                  { value: "none", label: "Clean" },
                  { value: "mask", label: "Mask" },
                  { value: "attenuate", label: "Attenuate" },
                  { value: "noise", label: "Noise" },
                ]}
                className="w-full [&>button]:flex-1"
              />
            </div>
            {(attack.type === "mask" || attack.type === "attenuate") && (
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <span className="eyebrow">{attack.type === "mask" ? "Suppressed share of window" : "Attenuation"}</span>
                  <span className="num text-[13px] font-medium">{Math.round(attack.severity * 100)}%</span>
                </div>
                <Slider value={attack.severity} onChange={(v) => setAttack({ severity: Math.round(v * 20) / 20 })} min={0.05} max={1} step={0.05} label="Severity" />
                <div className="mt-1.5 flex justify-between text-[10.5px] text-faint">
                  <span>5%</span>
                  <span>report grid 10–80%</span>
                  <span>100%</span>
                </div>
              </div>
            )}
            {attack.type === "mask" && (
              <div>
                <div className="eyebrow mb-2">Fill</div>
                <Segmented
                  value={attack.fill}
                  onChange={(f) => setAttack({ fill: f as Fill })}
                  options={[
                    { value: "zero", label: "Zero", title: "Sensor reports zeros" },
                    { value: "missing", label: "Missing", title: "Dropped samples (NaN)" },
                    { value: "hold", label: "Hold", title: "Frozen sensor (sample-and-hold)" },
                  ]}
                  className="w-full [&>button]:flex-1"
                />
              </div>
            )}
            {attack.type === "noise" && (
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <span className="eyebrow">Signal-to-noise ratio</span>
                  <span className="num text-[13px] font-medium">{attack.snr_db} dB</span>
                </div>
                <Slider value={attack.snr_db} onChange={(v) => setAttack({ snr_db: Math.round(v) })} min={-5} max={30} step={1} label="SNR" />
              </div>
            )}
            <div>
              <div className="eyebrow mb-2 flex items-center gap-1">
                Reliability evidence <InfoTip>Default (report design): signal quality Q, cross-modal disagreement D and predictive uncertainty U are combined into the reliability score. Switch a channel off to see its contribution; the reference distribution is recomputed for the remaining sum.</InfoTip>
              </div>
              <div className="flex gap-2">
                {["Q", "D", "U"].map((c) => (
                  <button
                    key={c}
                    onClick={() => toggleComponent(c)}
                    className={clsx(
                      "flex-1 rounded-lg border py-1.5 text-[12.5px] font-medium transition-colors",
                      components.includes(c) ? "border-defended/40 bg-defended-soft text-defended" : "border-line text-faint line-through",
                    )}
                  >
                    {c}
                  </button>
                ))}
              </div>
              <div className="eyebrow mb-2 mt-4 flex items-center gap-1">
                Calibration reference <InfoTip>Validation (report design): every scaling parameter and threshold comes from the clean validation subjects of this fold. Pooled (extension): the signal-quality reference pools all non-test subjects, which lowers false alarms on new people.</InfoTip>
              </div>
              <Segmented
                value={calibration}
                onChange={setCalibration}
                options={[
                  { value: "validation", label: "Validation", title: "Report design: validation subjects only" },
                  { value: "pooled", label: "Pooled", title: "Extension: signal-quality reference pooled over all non-test subjects" },
                ]}
              />
            </div>
            <div className="space-y-2 border-t border-line pt-4">
              <div className="eyebrow">Attacker strategies</div>
              <Button className="w-full" icon={<Crosshair className="h-3.5 w-3.5" />} loading={targeted.isPending} onClick={() => targeted.mutate()}>
                Find most damaging modality
              </Button>
              <Button className="w-full" icon={<ScanSearch className="h-3.5 w-3.5" />} loading={adaptive.isPending} onClick={() => adaptive.mutate()}>
                Run defense-aware attacker
              </Button>
            </div>
          </div>
        </Card>

        {/* ---------------- signals ---------------- */}
        <Card className="col-span-12 xl:col-span-5">
          <CardHeader
            eyebrow="Synchronized window"
            title={
              r ? (
                <span className="flex items-center gap-2">
                  {clock(r.window.start_s)}–{clock(r.window.end_s)} <Badge>true: {r.window.label_name}</Badge>
                </span>
              ) : (
                "Loading…"
              )
            }
            actions={attackOn && r ? <AttackBadge label={r.attack_label} /> : <Badge tone="good">clean</Badge>}
          />
          <div className="px-3 pb-3 pt-2">
            {infer.isError && <ErrorNote error={infer.error} />}
            {!r && !infer.isError && <Skeleton className="m-2 h-[380px]" />}
            {r &&
              mods.map((m) => (
                <div key={m} className="border-b border-line/70 py-1 last:border-0">
                  <div className="flex items-center justify-between px-2 pt-1 text-[11.5px]">
                    <span className="flex items-center gap-1.5 font-medium">
                      <span className="h-2 w-2 rounded-full" style={{ background: MODALITY_COLOR[m] }} />
                      {MODALITY_LABEL[m]}
                      <span className="font-normal text-muted">
                        {r.signals[m].unit} · {r.signals[m].hz} Hz
                      </span>
                    </span>
                    {attackOn && attack.modality === m && <span className="text-[11px] font-medium text-attack">suppressed</span>}
                  </div>
                  <SignalLane
                    modality={m}
                    hz={r.signals[m].hz}
                    unit={r.signals[m].unit}
                    original={r.signals[m].original}
                    attacked={r.signals[m].attacked}
                    channels={r.signals[m].channels}
                    attacked_on={attackOn && attack.modality === m}
                    t0={r.window.start_s}
                    height={mods.length > 2 ? 78 : 120}
                  />
                </div>
              ))}
          </div>
        </Card>

        {/* ---------------- decisions ---------------- */}
        <div className="col-span-12 space-y-4 xl:col-span-4">
          <Card>
            <CardHeader eyebrow="Decision" title="Fixed vs. reliability-aware fusion" actions={r && <FlagBadge on={r.flag} compact />} />
            {r ? (
              <div className="divide-y divide-line">
                {(["fixed", "defended"] as const).map((k) => (
                  <div key={k} className={clsx("px-5 py-4 transition-opacity", infer.isFetching && "opacity-70")}>
                    <div className="mb-2 flex items-center justify-between">
                      <span className="flex items-center gap-1.5 text-[12px] font-medium text-muted">
                        <span className={clsx("h-2 w-2 rounded-full", k === "fixed" ? "bg-fixed" : "bg-defended")} />
                        {k === "fixed" ? "Standard (fixed) fusion" : "Reliability-aware fusion"}
                      </span>
                      <span className="num text-[11.5px] text-muted">clean: {r.classes[r.clean[k].pred]}</span>
                    </div>
                    <Verdict head={r[k]} classes={r.classes} truth={r.window.label} changed={r.changed[k]} />
                    <div className="mt-3">
                      <ProbBars head={r[k]} classes={r.classes} tone={k} truth={r.window.label} />
                    </div>
                  </div>
                ))}
                <div className="flex items-start gap-2 bg-sunken/60 px-5 py-3 text-[12.5px] text-ink-2">
                  <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-muted" />
                  <span>{explain(r)}</span>
                </div>
              </div>
            ) : (
              <Skeleton className="m-5 h-64" />
            )}
          </Card>
        </div>

        {/* ---------------- reliability ---------------- */}
        <Card className="col-span-12 xl:col-span-7">
          <CardHeader eyebrow="Reliability engine" title="Per-modality evidence and fusion weights" actions={<EvidenceLegend />} />
          <div className="grid gap-2.5 p-4 sm:grid-cols-2">
            {r
              ? mods.map((m) => (
                  <ReliabilityRow key={m} m={m} ev={r.modalities[m]} threshold={r.flag_threshold} target={attackOn && attack.modality === m} components={r.components} />
                ))
              : mods.map((m) => <Skeleton key={m} className="h-[92px]" />)}
          </div>
          {r && (
            <div className="border-t border-line px-5 py-3">
              <div className="eyebrow mb-2">Unimodal head opinions (auxiliary heads of the same checkpoint)</div>
              <div className="grid gap-x-6 gap-y-1.5 sm:grid-cols-2">
                {mods.map((m) => {
                  const p = r.modalities[m].unimodal;
                  const pred = p.indexOf(Math.max(...p));
                  return (
                    <div key={m} className="flex items-center gap-2 text-[12px]">
                      <span className="w-10 font-medium" style={{ color: COLORS.ink }}>
                        {MODALITY_LABEL[m]}
                      </span>
                      <div className="flex h-2 flex-1 overflow-hidden rounded-full ring-1 ring-inset ring-line">
                        {p.map((v, i) => (
                          <div key={i} style={{ width: `${v * 100}%`, background: i === 0 ? "#d0d5dd" : i === 1 ? "#667085" : "#344054" }} />
                        ))}
                      </div>
                      <span className="w-24 truncate text-right text-muted">{r.classes[pred]}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </Card>
        <Card className="col-span-12 xl:col-span-5">
          <CardHeader eyebrow="Runtime sequence" title="What just happened (report Fig. 6, live)" />
          <div className="p-5">{r ? <SequenceTrace trace={r.trace} version={version} attacked={attackOn} /> : <Skeleton className="h-56" />}</div>
        </Card>

        {(targeted.data || targeted.isError) && (
          <Card className="col-span-12 xl:col-span-5">
            <CardHeader
              eyebrow="Attacker strategy"
              title="Which modality should an attacker suppress?"
              sub="Each modality fully suppressed in turn on this window."
              actions={<Button size="sm" variant="ghost" onClick={() => targeted.reset()}>Close</Button>}
            />
            {targeted.isError ? <div className="p-5"><ErrorNote error={targeted.error} /></div> : <TargetedTable data={targeted.data!} classes={r?.classes ?? []} onApply={(m) => setAttack({ type: "mask", modality: m, severity: 1, fill: targeted.data!.fill })} />}
          </Card>
        )}
        {(adaptive.data || adaptive.isError) && (
          <Card className={clsx("col-span-12", targeted.data ? "xl:col-span-7" : "")}>
            <CardHeader
              eyebrow="Defense-aware attacker (extension)"
              title="Searching for an evasive suppression"
              sub="Every candidate (modality × fill × severity, plus attenuation) evaluated against the defended pipeline."
              actions={<Button size="sm" variant="ghost" onClick={() => adaptive.reset()}>Close</Button>}
            />
            {adaptive.isError ? <div className="p-5"><ErrorNote error={adaptive.error} /></div> : <AdaptivePanel data={adaptive.data!} classes={r?.classes ?? []} onApply={(c) => setAttack(c.type === "mask" ? { type: "mask", modality: c.modality, severity: c.severity, fill: c.fill ?? "zero", placement: "center" } : { type: "attenuate", modality: c.modality, severity: c.severity })} />}
          </Card>
        )}
      </div>
      {nWin === 0 && windows.isSuccess && <Empty title="No labelled windows for this subject" />}
    </div>
  );
}


function explain(r: InferResult) {
  const truth = r.window.label;
  if (r.attack.type === "none") return "Clean window: both pipelines see identical evidence. The defense keeps every weight at 1 unless the evidence itself looks unusual.";
  const f = r.fixed.pred === truth;
  const d = r.defended.pred === truth;
  const tm = r.attack.modality ?? "";
  const m = MODALITY_LABEL[tm] ?? tm;
  if (!f && d) return `Suppressing ${m} flipped the fixed-fusion decision. The reliability engine scored ${m} as unreliable (R=${r.modalities[tm].R.toFixed(3)}) and down-weighted it, recovering the correct decision.`;
  if (!f && !d) return `Suppressing ${m} defeats both pipelines on this window${r.flag ? ", but the defended pipeline raises a low-confidence flag, so the error is not silent" : " and the evidence did not cross the flag threshold, so this is a stealthy success for the attacker"}.`;
  if (f && d) return r.flag ? `Both decisions survive the attack, and the defense still flags the suppressed ${m} evidence.` : `Both decisions survive this attack on ${m}; the remaining modalities carry enough evidence.`;
  return `The defense changed a correct decision here: an honest cost of down-weighting ${m}.`;
}

function TargetedTable({ data, classes, onApply }: { data: TargetedResult; classes: string[]; onApply: (m: string) => void }) {
  return (
    <div className="p-2">
      <table className="w-full text-[12.5px]">
        <thead>
          <tr className="text-left text-[11px] uppercase tracking-wide text-muted">
            <th className="px-3 py-2 font-medium">Suppress</th>
            <th className="px-3 py-2 font-medium">Output shift</th>
            <th className="px-3 py-2 font-medium">Fixed</th>
            <th className="px-3 py-2 font-medium">Defended</th>
            <th className="px-3 py-2 font-medium" />
          </tr>
        </thead>
        <tbody>
          {data.candidates.map((c) => (
            <tr key={c.modality} className="border-t border-line">
              <td className="px-3 py-2">
                <span className="flex items-center gap-1.5 font-medium">
                  <span className="h-2 w-2 rounded-full" style={{ background: MODALITY_COLOR[c.modality] }} />
                  {MODALITY_LABEL[c.modality]}
                  {c.modality === data.targeted && <Badge tone="attack">targeted pick</Badge>}
                  {c.modality === data.oracle && c.modality !== data.targeted && <Badge>oracle pick</Badge>}
                </span>
              </td>
              <td className="num px-3 py-2">{c.tv.toFixed(3)}</td>
              <td className={clsx("px-3 py-2", c.fixed.pred !== data.label && "font-medium text-attack")}>{classes[c.fixed.pred]}</td>
              <td className={clsx("px-3 py-2", c.defended.pred !== data.label ? "font-medium text-attack" : "text-good-ink")}>
                {classes[c.defended.pred]} {c.flag && <span className="text-flag-ink">⚑</span>}
              </td>
              <td className="px-3 py-2 text-right">
                <Button size="sm" variant="ghost" onClick={() => onApply(c.modality)}>
                  Apply
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="px-3 pb-2 pt-1 text-[11.5px] text-muted">
        Targeted = label-free attacker querying the model and picking the largest output shift (total variation). Oracle = label-aware worst case. ⚑ = flag raised.
      </p>
    </div>
  );
}

function AdaptivePanel({ data, classes, onApply }: { data: AdaptiveResult; classes: string[]; onApply: (c: AdaptiveResult["candidates"][number]) => void }) {
  const cleanDef = data.clean.defended.pred;
  const option = useMemo(() => {
    const groups = [
      { name: "Evasive flip (attack succeeds)", color: COLORS.attack, f: (c: any) => c.defended_pred !== cleanDef && !c.flag },
      { name: "Flip but flagged", color: COLORS.flag, f: (c: any) => c.defended_pred !== cleanDef && c.flag },
      { name: "No flip", color: "#98a2b3", f: (c: any) => c.defended_pred === cleanDef },
    ];
    const mods = Array.from(new Set(data.candidates.map((c) => c.modality)));
    return {
      grid: { left: 8, right: 16, top: 34, bottom: 28 },
      legend: { top: 0, left: 0, icon: "circle", itemWidth: 8, itemHeight: 8 },
      tooltip: {
        trigger: "item",
        formatter: (p: any) => {
          const c = p.data.c;
          return (
            ttTitle(`${MODALITY_LABEL[c.modality]} · ${c.type === "mask" ? `mask (${c.fill})` : "attenuate"} ${Math.round(c.severity * 100)}%`) +
            ttRow(COLORS.defended, "defended", classes[c.defended_pred] ?? String(c.defended_pred)) +
            ttRow(COLORS.fixed, "fixed", classes[c.fixed_pred] ?? String(c.fixed_pred)) +
            ttRow("#fff", "anomaly margin", c.suspicion.toFixed(2)) +
            ttRow("#fff", "flag", c.flag ? "raised" : "no")
          );
        },
      },
      xAxis: { type: "value", name: "severity", nameLocation: "middle", nameGap: 20, min: 0, max: 1.05, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` } },
      yAxis: { type: "category", data: mods.map((m) => MODALITY_LABEL[m]), axisLabel: { color: COLORS.ink2 } },
      series: groups.map((g) => ({
        name: g.name,
        type: "scatter",
        symbolSize: (v: any) => 7 + Math.max(0, Math.min(v[2], 8)),
        itemStyle: { color: g.color, borderColor: "#fff", borderWidth: 1 },
        data: data.candidates
          .filter(g.f)
          .map((c) => ({ value: [c.severity + (c.type === "attenuate" ? 0.012 : c.fill === "missing" ? -0.012 : 0), mods.indexOf(c.modality), c.suspicion], c })),
      })),
    };
  }, [data, cleanDef, classes]);
  return (
    <div className="grid gap-4 p-5 lg:grid-cols-[1fr_260px]">
      <EChart option={option} height={230} onEvents={{ click: (p: any) => p.data?.c && onApply(p.data.c) }} />
      <div className="space-y-3 text-[12.5px]">
        <div className={clsx("rounded-lg border p-3", data.evasive_success ? "border-attack/30 bg-attack-soft" : "border-good/30 bg-good-soft")}>
          <div className="flex items-center gap-1.5 font-medium">
            <Sparkles className="h-3.5 w-3.5" />
            {data.evasive_success ? "Evasive attack found" : "No evasive attack on this window"}
          </div>
          <p className="mt-1 text-ink-2">
            {data.evasive_success && data.chosen
              ? `Cheapest evasion: ${MODALITY_LABEL[data.chosen.modality]} ${data.chosen.type === "mask" ? `${data.chosen.fill}-mask` : "attenuation"} at ${Math.round(data.chosen.severity * 100)}%, flipping the defended decision without raising the flag.`
              : "Every candidate that flips the defended decision also raises the low-confidence flag, or none flips it."}
          </p>
        </div>
        <dl className="grid grid-cols-2 gap-2 text-muted">
          <dt>Candidates</dt>
          <dd className="num text-right text-ink">{data.n_candidates}</dd>
          <dt>Flip fixed fusion</dt>
          <dd className="num text-right text-ink">{data.fixed_flippable}</dd>
          <dt>Flagged</dt>
          <dd className="num text-right text-ink">{data.flagged}</dd>
        </dl>
        {data.chosen && (
          <Button size="sm" className="w-full" onClick={() => onApply(data.chosen!)}>
            Apply chosen attack
          </Button>
        )}
        <p className="text-[11.5px] text-muted">Click any point to load that attack into the lab.</p>
      </div>
    </div>
  );
}

