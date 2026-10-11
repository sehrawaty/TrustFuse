import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { CheckCircle2, CircleDot, Clock, FlaskConical, Loader2, Play, Square, XCircle } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import EChart from "../components/charts/EChart";
import { Badge, Button, Card, CardHeader, Empty, ErrorNote, InfoTip, PageHeader, Segmented, Skeleton } from "../components/ui";
import { api, type JobEvent, type JobPublic, type Preset } from "../lib/api";
import { ago, seconds } from "../lib/format";
import { COLORS } from "../lib/palette";
import { useApp } from "../lib/store";

const ALL_SUBJECTS: Record<string, string[]> = {
  wesad: ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"],
  dalia: Array.from({ length: 15 }, (_, i) => `S${i + 1}`),
};
const STRATEGIES = ["random_drop", "targeted", "oracle", "adaptive"];
const SEVERITIES = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0];

export default function Experiments() {
  const qc = useQueryClient();
  const presets = useQuery({ queryKey: ["presets"], queryFn: api.presets });
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 4000 });
  const [presetKey, setPresetKey] = useState("quick_demo");
  const preset = presets.data?.find((p) => p.key === presetKey);
  const [over, setOver] = useState<Record<string, any>>({});
  useEffect(() => setOver({}), [presetKey]);
  const cfg = useMemo(() => (preset ? deepMerge(preset.config, over) : null), [preset, over]);
  const overrides = useMemo(() => (preset ? diffOverrides(preset.config, cfg!) : {}), [preset, cfg]);
  const validation = useQuery({
    queryKey: ["validate", presetKey, overrides],
    queryFn: () => api.validate({ preset: presetKey, overrides }),
    enabled: !!preset,
    retry: 0,
  });
  const submit = useMutation({
    mutationFn: () => api.submit({ preset: presetKey, overrides }),
    onSuccess: (j) => {
      setActive(j.id);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const running = jobs.data?.find((j) => j.status === "running" || j.status === "queued");
  const [active, setActive] = useState<string | null>(null);
  const shown = active ?? running?.id ?? jobs.data?.[0]?.id ?? null;
  const dataset = cfg?.task?.startsWith("dalia") ? "dalia" : "wesad";

  return (
    <div>
      <PageHeader
        eyebrow="Evaluate"
        title="Experiments"
        description="Every result in this application can be reproduced from here. Runs are versioned by configuration hash, executed one at a time inside the API container, and written to a separate runtime area, so the shipped canonical results are never overwritten."
      />
      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-12 space-y-4 xl:col-span-5">
          <Card>
            <CardHeader eyebrow="1 · Preset" title="Choose an experiment" />
            <div className="space-y-2 p-4">
              {presets.isLoading && <Skeleton className="h-48" />}
              {presets.data?.map((p) => (
                <button
                  key={p.key}
                  onClick={() => setPresetKey(p.key)}
                  className={clsx(
                    "w-full rounded-lg border px-3.5 py-2.5 text-left transition-all",
                    presetKey === p.key ? "border-ink bg-surface ring-1 ring-ink" : "border-line hover:border-line-strong",
                  )}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[13px] font-semibold">{p.label}</span>
                    <span className={clsx("flex items-center gap-1 text-[11.5px]", p.minutes > 30 ? "text-flag-ink" : "text-muted")}>
                      <Clock className="h-3 w-3" /> ~{p.minutes >= 60 ? `${Math.round(p.minutes / 60)} h` : `${p.minutes} min`}
                    </span>
                  </div>
                  <div className="mt-0.5 text-[12px] text-muted">{p.summary}</div>
                </button>
              ))}
            </div>
          </Card>
          {cfg && preset && (
            <Card>
              <CardHeader eyebrow="2 · Configure" title={cfg.name} sub={cfg.description} />
              <ConfigEditor cfg={cfg} preset={preset} dataset={dataset} onChange={(patch) => setOver((o) => deepMerge(o, patch))} />
              <div className="border-t border-line px-4 py-3">
                {validation.isError ? (
                  <ErrorNote error={validation.error} />
                ) : (
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-[12px] text-muted">
                      config hash <span className="num text-ink">{validation.data?.config_hash ?? "…"}</span> · {validation.data?.train_jobs ?? "…"} models
                      {cfg.models_root === "artifacts" && " (frozen, already trained)"}
                    </div>
                    <Button variant="primary" icon={<Play className="h-3.5 w-3.5" />} loading={submit.isPending} disabled={!!running || validation.isError} onClick={() => submit.mutate()}>
                      {running ? "A job is running" : "Launch"}
                    </Button>
                  </div>
                )}
                {submit.isError && <div className="mt-2"><ErrorNote error={submit.error} /></div>}
                {preset.minutes > 30 && (
                  <p className="mt-2 text-[11.5px] text-flag-ink">This is the full experiment and takes hours on a laptop. The shipped canonical results were produced with exactly this configuration.</p>
                )}
              </div>
              {validation.data?.yaml && (
                <details className="border-t border-line px-4 py-3 text-[12px]">
                  <summary className="cursor-pointer text-muted hover:text-ink">Show the resolved YAML configuration</summary>
                  <pre className="mt-2 max-h-72 overflow-auto rounded-lg bg-sunken p-3 font-mono text-[11px] leading-relaxed text-ink-2 ring-1 ring-line">{validation.data.yaml}</pre>
                </details>
              )}
            </Card>
          )}
        </div>
        <div className="col-span-12 space-y-4 xl:col-span-7">
          {shown ? <JobMonitor id={shown} /> : <Card><Empty title="No experiment has been run in this session" icon={<FlaskConical className="h-6 w-6" />}>Launch the quick demo to watch training, calibration and the attack matrix run live.</Empty></Card>}
          <Card>
            <CardHeader eyebrow="History" title="Runs in this session" sub={"Each .\\run starts a fresh session: user runs are cleared, shipped results stay."} />
            {jobs.data?.length ? (
              <table className="w-full text-[12.5px]">
                <tbody>
                  {jobs.data.map((j) => (
                    <tr key={j.id} onClick={() => setActive(j.id)} className={clsx("cursor-pointer border-t border-line first:border-0 hover:bg-sunken", shown === j.id && "bg-sunken")}>
                      <td className="px-4 py-2">
                        <StatusIcon s={j.status} />
                      </td>
                      <td className="px-2 py-2 font-medium">{j.name}</td>
                      <td className="px-2 py-2 text-muted">{j.task}</td>
                      <td className="num px-2 py-2 text-muted">{ago(j.created)}</td>
                      <td className="num px-4 py-2 text-right text-muted">{j.finished && j.started ? seconds(j.finished - j.started) : `${Math.round(j.progress * 100)}%`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="px-4 py-6 text-center text-[12.5px] text-muted">No runs yet.</div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}

function StatusIcon({ s }: { s: JobPublic["status"] }) {
  if (s === "completed") return <CheckCircle2 className="h-4 w-4 text-good" />;
  if (s === "failed") return <XCircle className="h-4 w-4 text-attack" />;
  if (s === "cancelled") return <Square className="h-4 w-4 text-faint" />;
  if (s === "running") return <Loader2 className="h-4 w-4 animate-spin text-defended" />;
  return <CircleDot className="h-4 w-4 text-faint" />;
}

function ConfigEditor({ cfg, preset, dataset, onChange }: { cfg: any; preset: Preset; dataset: string; onChange: (p: any) => void }) {
  const subjects = ALL_SUBJECTS[dataset];
  const testSubjects: string[] = cfg.test_subjects ?? subjects;
  const toggle = (arr: any[], v: any) => (arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v]);
  const frozen = cfg.models_root === "artifacts";
  return (
    <div className="space-y-4 p-4 text-[12.5px]">
      <Field label="Held-out test subjects" hint="Each selected subject is one LOSO fold: trained without it, validated on two other subjects, then attacked.">
        <div className="flex flex-wrap gap-1.5">
          {subjects.map((s) => (
            <button
              key={s}
              onClick={() => {
                const next = toggle(testSubjects, s);
                if (next.length) onChange({ test_subjects: subjects.filter((x) => next.includes(x)) });
              }}
              className={clsx("num rounded-md border px-2 py-0.5 text-[11.5px]", testSubjects.includes(s) ? "border-ink bg-ink text-white" : "border-line text-muted hover:border-line-strong")}
            >
              {s}
            </button>
          ))}
        </div>
      </Field>
      {!frozen && (
        <div className="grid grid-cols-2 gap-4">
          <Field label="Max epochs">
            <Segmented size="sm" value={cfg.train.epochs} onChange={(v) => onChange({ train: { epochs: v } })} options={[4, 8, 15, 40].map((v) => ({ value: v, label: String(v) }))} />
          </Field>
          <Field label="Seeds (fusion)">
            <Segmented size="sm" value={cfg.seeds.length} onChange={(v) => onChange({ seeds: Array.from({ length: v }, (_, i) => i), eval_seeds: Array.from({ length: v }, (_, i) => i) })} options={[1, 2, 3].map((v) => ({ value: v, label: String(v) }))} />
          </Field>
        </div>
      )}
      <Field label="Masking severities" hint="Share of the window suppressed (the report's grid is 10–80%, plus full masking).">
        <div className="flex flex-wrap gap-1.5">
          {SEVERITIES.map((s) => (
            <button
              key={s}
              onClick={() => {
                const next = toggle(cfg.attacks.severities, s).sort((a: number, b: number) => a - b);
                if (next.length) onChange({ attacks: { severities: next } });
              }}
              className={clsx("num rounded-md border px-2 py-0.5 text-[11.5px]", cfg.attacks.severities.includes(s) ? "border-ink bg-ink text-white" : "border-line text-muted")}
            >
              {Math.round(s * 100)}%
            </button>
          ))}
        </div>
      </Field>
      <Field label="Attacker strategies">
        <div className="flex flex-wrap gap-1.5">
          {STRATEGIES.map((s) => (
            <button
              key={s}
              onClick={() => onChange({ attacks: { strategies: toggle(cfg.attacks.strategies, s) } })}
              className={clsx("rounded-md border px-2 py-0.5 text-[11.5px]", cfg.attacks.strategies.includes(s) ? "border-ink bg-ink text-white" : "border-line text-muted")}
            >
              {s.replace("_", " ")}
            </button>
          ))}
        </div>
      </Field>
      <div className="grid grid-cols-2 gap-4">
        <Field label="Flag level (alpha)" hint="Target clean false-rejection rate of the low-confidence flag.">
          <Segmented size="sm" value={cfg.reliability.alpha} onChange={(v) => onChange({ reliability: { alpha: v } })} options={[0.01, 0.05, 0.1].map((v) => ({ value: v, label: String(v) }))} />
        </Field>
        <Field label="Evidence">
          <div className="flex gap-1.5">
            {["Q", "D", "U"].map((c) => (
              <button
                key={c}
                onClick={() => {
                  const next = toggle(cfg.reliability.components, c).sort((a: string, b: string) => "QDU".indexOf(a) - "QDU".indexOf(b));
                  if (next.length) onChange({ reliability: { components: next } });
                }}
                className={clsx("w-8 rounded-md border py-0.5 text-[11.5px] font-medium", cfg.reliability.components.includes(c) ? "border-defended/40 bg-defended-soft text-defended" : "border-line text-faint line-through")}
              >
                {c}
              </button>
            ))}
          </div>
        </Field>
      </div>
      {frozen && <p className="text-[11.5px] text-muted">Uses the shipped fold models ({preset.config.task}); only calibration and the attack matrix are recomputed.</p>}
    </div>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="eyebrow mb-1.5 flex items-center gap-1">
        {label}
        {hint && <InfoTip>{hint}</InfoTip>}
      </div>
      {children}
    </div>
  );
}

const STAGES = ["queued", "training", "evaluating", "completed"];

function JobMonitor({ id }: { id: string }) {
  const qc = useQueryClient();
  const { setRun, setTask } = useApp();
  const navigate = useNavigate();
  const initial = useQuery({ queryKey: ["job", id], queryFn: () => api.job(id) });
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [job, setJob] = useState<JobPublic | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const cancel = useMutation({ mutationFn: () => api.cancel(id) });

  useEffect(() => {
    setEvents([]);
    setJob(null);
    const es = new EventSource(`/api/jobs/${id}/events`);
    const handler = (e: MessageEvent) => {
      const d = JSON.parse(e.data);
      if (d.job) setJob(d.job);
      if (d.seq != null) setEvents((prev) => (prev.length && prev[prev.length - 1].seq >= d.seq ? prev : [...prev, d]));
    };
    ["log", "progress", "epoch", "stage", "status"].forEach((t) => es.addEventListener(t, handler as any));
    es.addEventListener("end", (e: MessageEvent) => {
      const d = JSON.parse(e.data);
      setJob(d.job);
      es.close();
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["results"] });
    });
    return () => es.close();
  }, [id, qc]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [events.length]);

  const j = job ?? initial.data ?? null;
  const epochs = events.filter((e) => e.type === "epoch");
  const logs = events.filter((e) => e.type === "log");
  const currentModel = [...events].reverse().find((e) => e.type === "stage" && e.data.model)?.data.model;
  const chart = useMemo(() => {
    const byModel = new Map<string, { epoch: number; val_f1: number; loss: number }[]>();
    epochs.forEach((e) => {
      const k = e.data.model as string;
      if (!byModel.has(k)) byModel.set(k, []);
      byModel.get(k)!.push(e.data as any);
    });
    const last = Array.from(byModel.entries()).slice(-4);
    return {
      grid: { left: 8, right: 12, top: 34, bottom: 24 },
      legend: { top: 0, left: 0, type: "scroll" },
      tooltip: { trigger: "axis" },
      xAxis: { type: "value", name: "epoch", nameLocation: "middle", nameGap: 20, minInterval: 1 },
      yAxis: { type: "value", min: 0, max: 1, name: "val macro-F1", nameTextStyle: { align: "left" } },
      series: last.map(([k, pts], i) => ({
        type: "line",
        name: k,
        data: pts.map((p) => [p.epoch, p.val_f1]),
        symbolSize: 5,
        lineStyle: { width: 2, color: i === last.length - 1 ? COLORS.defended : "#c4c9d2" },
        itemStyle: { color: i === last.length - 1 ? COLORS.defended : "#c4c9d2" },
      })),
    };
  }, [epochs]);

  if (!j) return <Skeleton className="h-96" />;
  const stageIdx = j.status === "completed" ? 3 : STAGES.indexOf(j.stage);
  return (
    <Card>
      <CardHeader
        eyebrow="Live run"
        title={j.name}
        sub={`job ${j.id} · ${j.task}`}
        actions={
          j.status === "running" || j.status === "queued" ? (
            <Button size="sm" variant="danger" loading={cancel.isPending} onClick={() => cancel.mutate()}>
              Cancel
            </Button>
          ) : j.status === "completed" && j.run_id ? (
            <Button
              size="sm"
              variant="primary"
              onClick={() => {
                setTask(j.task);
                setRun(j.task, j.run_id!);
                navigate("/results");
              }}
            >
              Open results
            </Button>
          ) : (
            <Badge tone={j.status === "failed" ? "attack" : "neutral"}>{j.status}</Badge>
          )
        }
      />
      <div className="p-5">
        <div className="mb-4 flex items-center gap-2">
          {STAGES.map((s, i) => (
            <div key={s} className="flex flex-1 items-center gap-2">
              <div className={clsx("flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-medium ring-1", i < stageIdx || j.status === "completed" ? "bg-ink text-white ring-ink" : i === stageIdx ? "bg-defended-soft text-defended ring-defended/40" : "bg-surface text-faint ring-line")}>
                {i < stageIdx || j.status === "completed" ? "✓" : i + 1}
              </div>
              <span className={clsx("text-[12px] capitalize", i === stageIdx ? "font-medium text-ink" : "text-muted")}>{s}</span>
              {i < STAGES.length - 1 && <div className="h-px flex-1 bg-line" />}
            </div>
          ))}
        </div>
        <div className="mb-1 flex justify-between text-[12px]">
          <span className="text-muted">{currentModel ? `training ${currentModel}` : j.stage}</span>
          <span className="num">{Math.round(j.progress * 100)}%</span>
        </div>
        <div className="h-1.5 overflow-hidden rounded-full bg-line">
          <div className={clsx("h-full transition-all duration-500", j.status === "failed" ? "bg-attack" : "bg-ink")} style={{ width: `${Math.max(2, j.progress * 100)}%` }} />
        </div>
        {j.error && <div className="mt-3"><ErrorNote error={j.error} /></div>}
        {epochs.length > 0 && (
          <div className="mt-5">
            <EChart option={chart} height={200} />
          </div>
        )}
        <div ref={logRef} className="mt-4 h-52 overflow-auto rounded-lg bg-[#0f1420] p-3 font-mono text-[11px] leading-relaxed text-[#c9d1dd]">
          {logs.length === 0 && <span className="text-[#6b7686]">waiting for output…</span>}
          {logs.map((l) => (
            <div key={l.seq} className="whitespace-pre-wrap">
              <span className="text-[#6b7686]">{new Date(l.t * 1000).toLocaleTimeString()} </span>
              {String(l.data.message)}
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}

function deepMerge(a: any, b: any): any {
  if (Array.isArray(b) || typeof b !== "object" || b === null) return b;
  const out: any = { ...(a ?? {}) };
  for (const [k, v] of Object.entries(b)) out[k] = v && typeof v === "object" && !Array.isArray(v) ? deepMerge(out[k], v) : v;
  return out;
}

function diffOverrides(base: any, next: any): any {
  const out: any = {};
  for (const [k, v] of Object.entries(next)) {
    if (JSON.stringify(v) === JSON.stringify(base?.[k])) continue;
    out[k] = v && typeof v === "object" && !Array.isArray(v) ? diffOverrides(base?.[k], v) : v;
  }
  return out;
}
