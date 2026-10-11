import * as Tabs from "@radix-ui/react-tabs";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { Download, FileJson } from "lucide-react";
import { useMemo, useState } from "react";
import EChart, { MONO_FONT, ttRow, ttTitle } from "../components/charts/EChart";
import { HypothesisCards } from "../components/Hypotheses";
import { Badge, Card, CardHeader, CiText, Empty, InfoTip, PageHeader, Segmented, Select, Skeleton } from "../components/ui";
import { useConditions, useDetection, useRun, useSummary } from "../components/useRun";
import { api, downloadUrl, type CI, type ConditionAgg, type Summary } from "../lib/api";
import { pval, signed } from "../lib/format";
import { BLUE_RAMP, COLORS, MODALITY_COLOR, MODALITY_LABEL } from "../lib/palette";

type M = { mean: number | null; lo: number | null; hi: number | null };
const fmt2 = (v: number) => (Math.abs(v) < 0.005 ? "0.00" : v.toFixed(2));
const mm = (c: ConditionAgg | undefined, k: string): M => ((c?.[k] as M) ?? { mean: null, lo: null, hi: null });
const VARIANT_LABEL: Record<string, string> = {
  fusion: "Fixed fusion",
  "fusion+defense": "Reliability-aware fusion",
  moddrop: "ModDrop fusion",
  "moddrop+defense": "ModDrop + reliability-aware",
  uni_bvp: "Unimodal · BVP",
  uni_acc: "Unimodal · ACC",
  uni_eda: "Unimodal · EDA",
  uni_temp: "Unimodal · TEMP",
};

export default function Results() {
  const { runId, runs, setRun, loading } = useRun();
  const summary = useSummary(runId);
  const conds = useConditions(runId);
  const det = useDetection(runId);
  const meta = runs.find((r) => r.run_id === runId);
  const s = summary.data;
  const fusion = useMemo(() => (conds.data?.conditions ?? []).filter((c) => c.variant === "fusion"), [conds.data]);

  return (
    <div>
      <PageHeader
        eyebrow="Evaluate"
        title="Security evaluation"
        description="Measured results only. Every value is a subject-level mean over leave-one-subject-out folds with 95% subject-bootstrap confidence intervals; seeds are averaged within each subject first."
        actions={
          <>
            {runs.length > 0 && (
              <Select
                value={runId ?? ""}
                onChange={setRun}
                options={runs.map((r) => ({ value: r.run_id, label: r.canonical ? `${r.run_id} · canonical` : r.run_id }))}
                className="w-[260px]"
              />
            )}
            {runId && (
              <a href={downloadUrl.bundle(runId)} className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-line-strong bg-surface px-3 text-[13px] font-medium hover:bg-sunken">
                <Download className="h-3.5 w-3.5" /> Result bundle
              </a>
            )}
          </>
        }
      />
      {!loading && !runId && (
        <Card>
          <Empty title="No result set for this task yet">Run an experiment from the Experiments page, or switch to a task with canonical results.</Empty>
        </Card>
      )}
      {meta && (
        <div className="mb-5 flex flex-wrap items-center gap-x-5 gap-y-1.5 rounded-xl border border-line bg-surface px-4 py-2.5 text-[12px] text-muted">
          <span className="flex items-center gap-1.5">
            <FileJson className="h-3.5 w-3.5" /> {meta.name}
          </span>
          {meta.canonical ? <Badge tone="dark">canonical</Badge> : <Badge>user run</Badge>}
          <span>
            config <span className="num text-ink">{meta.config_hash}</span>
          </span>
          <span>
            subjects <span className="num text-ink">{meta.subjects.length}</span>
          </span>
          <span>
            seeds <span className="num text-ink">{meta.seeds.join(", ")}</span>
          </span>
          <span>
            measurements <span className="num text-ink">{meta.n_rows.toLocaleString()}</span>
          </span>
          {meta.git_commit && (
            <span>
              commit <span className="num text-ink">{meta.git_commit.slice(0, 8)}</span>
            </span>
          )}
          <span className="ml-auto">{new Date(meta.created).toLocaleString()}</span>
        </div>
      )}
      {runId && (
        <Tabs.Root defaultValue="summary">
          <Tabs.List className="mb-5 flex gap-1 border-b border-line">
            {[
              ["summary", "Summary"],
              ["vulnerability", "Vulnerability"],
              ["defense", "Defense"],
              ["detection", "Detection"],
              ["attackers", "Attackers"],
              ["bursts", "Bursts"],
              ["ablations", "Ablations"],
              ["manifest", "Manifest"],
            ].map(([v, l]) => (
              <Tabs.Trigger
                key={v}
                value={v}
                className="-mb-px border-b-2 border-transparent px-3 py-2 text-[13px] font-medium text-muted transition-colors hover:text-ink data-[state=active]:border-ink data-[state=active]:text-ink"
              >
                {l}
              </Tabs.Trigger>
            ))}
          </Tabs.List>
          {!s || !conds.data ? (
            <Skeleton className="h-96" />
          ) : (
            <>
              <Tabs.Content value="summary">
                <SummaryTab s={s} />
              </Tabs.Content>
              <Tabs.Content value="vulnerability">
                <VulnerabilityTab s={s} fusion={fusion} />
              </Tabs.Content>
              <Tabs.Content value="defense">
                <DefenseTab s={s} fusion={fusion} />
              </Tabs.Content>
              <Tabs.Content value="detection">
                <DetectionTab s={s} det={det.data} fusion={fusion} />
              </Tabs.Content>
              <Tabs.Content value="attackers">
                <AttackersTab s={s} />
              </Tabs.Content>
              <Tabs.Content value="bursts">
                <BurstsTab s={s} fusion={fusion} />
              </Tabs.Content>
              <Tabs.Content value="ablations">
                <AblationsTab s={s} fusion={fusion} />
              </Tabs.Content>
              <Tabs.Content value="manifest">
                <ManifestTab runId={runId} classes={s.classes} />
              </Tabs.Content>
            </>
          )}
        </Tabs.Root>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ summary */

function SummaryTab({ s }: { s: Summary }) {
  const rows = s.baselines;
  const option = useMemo(() => {
    const order = [...rows].reverse();
    const color = (v: string) => (v.includes("defense") ? COLORS.defended : v === "fusion" || v === "moddrop" ? COLORS.fixed : MODALITY_COLOR[v.replace("uni_", "")] ?? COLORS.muted);
    return {
      grid: { left: 8, right: 30, top: 8, bottom: 28 },
      tooltip: {
        trigger: "item",
        formatter: (p: any) => {
          const r = order[p.dataIndex];
          return ttTitle(VARIANT_LABEL[r.variant] ?? r.variant) + ttRow(color(r.variant), "macro-F1", `${r.macro_f1.mean?.toFixed(3)} [${r.macro_f1.lo?.toFixed(3)}, ${r.macro_f1.hi?.toFixed(3)}]`);
        },
      },
      xAxis: { type: "value", min: (v: any) => Math.max(0, Math.floor(v.min * 10) / 10 - 0.05), max: 1, name: "clean macro-F1", nameLocation: "middle", nameGap: 22 },
      yAxis: { type: "category", data: order.map((r) => VARIANT_LABEL[r.variant] ?? r.variant), axisLabel: { color: COLORS.ink2 } },
      series: [
        {
          type: "custom",
          data: order.map((r, i) => [i, r.macro_f1.lo, r.macro_f1.hi]),
          renderItem: (_: any, api: any) => {
            const a = api.coord([api.value(1), api.value(0)]);
            const b = api.coord([api.value(2), api.value(0)]);
            return { type: "line", shape: { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, style: { stroke: COLORS.faint, lineWidth: 2 } };
          },
          silent: true,
        },
        {
          type: "scatter",
          data: order.map((r) => ({ value: [r.macro_f1.mean, VARIANT_LABEL[r.variant] ?? r.variant], itemStyle: { color: color(r.variant) } })),
          symbolSize: 11,
          itemStyle: { borderColor: "#fff", borderWidth: 1.5 },
        },
      ],
    };
  }, [rows]);
  const tvr = s.hypotheses.secondary.targeted_vs_random;
  return (
    <div className="space-y-4">
      <HypothesisCards s={s} />
      <div className="grid grid-cols-12 gap-4">
        <Card className="col-span-12 xl:col-span-7">
          <CardHeader eyebrow="Baselines" title="Clean subject-independent performance" sub={`${s.n_subjects} LOSO folds · ${s.n_windows.toLocaleString()} test windows`} />
          <div className="p-3">
            <EChart option={option} height={46 + rows.length * 30} />
          </div>
        </Card>
        <Card className="col-span-12 xl:col-span-5">
          <CardHeader eyebrow="Core claim" title="Strategic vs. random suppression" sub="Same budget: one modality removed per window." />
          <div className="space-y-4 p-5 text-[13px]">
            {tvr ? (
              <>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="eyebrow mb-1">Random modality loss</div>
                    <div className="text-[20px] font-semibold">{tvr.random_f1.mean?.toFixed(3)}</div>
                    <div className="text-xs text-muted">fixed-fusion macro-F1</div>
                  </div>
                  <div>
                    <div className="eyebrow mb-1">Targeted suppression</div>
                    <div className="text-[20px] font-semibold text-attack">{tvr.targeted_f1.mean?.toFixed(3)}</div>
                    <div className="text-xs text-muted">fixed-fusion macro-F1</div>
                  </div>
                </div>
                <p className="text-ink-2">
                  {tvr.statement} Difference <CiText ci={tvr.difference} />, one-sided Wilcoxon p {pval(tvr.p)}
                  {tvr.rank_biserial != null && <>, rank-biserial r = {tvr.rank_biserial.toFixed(2)}</>}.
                </p>
                <p className="text-[12px] text-muted">Robustness to accidental missingness therefore understates the security risk: the attacker's choice of modality matters.</p>
              </>
            ) : (
              <span className="text-muted">Not part of this run.</span>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ vulnerability */

function VulnerabilityTab({ s, fusion }: { s: Summary; fusion: ConditionAgg[] }) {
  const [fill, setFill] = useState<string>("zero");
  const [metric, setMetric] = useState<string>("drop");
  const [pipeline, setPipeline] = useState<"fixed" | "def">("fixed");
  const clean = fusion.find((c) => c.family === "clean");
  const mods = s.modalities;
  const sevs = Array.from(new Set(fusion.filter((c) => c.family === "mask").map((c) => c.severity as number))).sort((a, b) => a - b);
  const fills = Array.from(new Set(fusion.filter((c) => c.family === "mask").map((c) => c.fill as string)));
  const find = (fam: string, m: string, sev?: number, f?: string) =>
    fusion.find((c) => c.family === fam && c.modality === m && (sev == null || Math.abs((c.severity ?? -1) - sev) < 1e-9) && (f == null || c.fill === f));

  const heat = useMemo(() => {
    const key = (c?: ConditionAgg) => {
      if (!c) return null;
      if (metric === "drop") return (mm(clean, `${pipeline}_f1`).mean ?? 0) - (mm(c, `${pipeline}_f1`).mean ?? 0);
      if (metric === "asr") return mm(c, `${pipeline}_asr`).mean;
      return mm(c, `${pipeline}_cwrong`).mean;
    };
    const data: [number, number, number | null][] = [];
    mods.forEach((m, i) => sevs.forEach((sv, j) => data.push([j, i, key(find("mask", m, sv, fill))])));
    const vals = data.map((d) => d[2] ?? 0);
    return {
      grid: { left: 8, right: 70, top: 8, bottom: 30 },
      tooltip: { formatter: (p: any) => ttTitle(`${MODALITY_LABEL[mods[p.value[1]]]} · ${Math.round(sevs[p.value[0]] * 100)}% ${fill}`) + ttRow(BLUE_RAMP[4], metric === "drop" ? "macro-F1 drop" : metric === "asr" ? "attack success rate" : "confidently wrong", (p.value[2] ?? 0).toFixed(3)) },
      xAxis: { type: "category", data: sevs.map((v) => `${Math.round(v * 100)}%`), name: "suppressed share", nameLocation: "middle", nameGap: 22, splitArea: { show: false } },
      yAxis: { type: "category", data: mods.map((m) => MODALITY_LABEL[m]), axisLabel: { color: COLORS.ink2 } },
      visualMap: { min: 0, max: Math.max(0.05, ...vals), calculable: false, orient: "vertical", right: 0, top: "middle", itemHeight: 120, inRange: { color: BLUE_RAMP }, textStyle: { fontSize: 10.5, color: COLORS.muted, fontFamily: MONO_FONT }, formatter: (v: number) => v.toFixed(2) },
      series: [{ type: "heatmap", data: data.map((d) => ({ value: d, label: { color: (d[2] ?? 0) > 0.55 * Math.max(0.05, ...vals) ? "#fff" : COLORS.ink } })), label: { show: true, fontSize: 11, fontFamily: MONO_FONT, formatter: (p: any) => (p.value[2] == null ? "" : fmt2(p.value[2])) }, itemStyle: { borderColor: "#fff", borderWidth: 2, borderRadius: 4 } }],
    };
  }, [fusion, fill, metric, pipeline, mods, sevs, clean]);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          eyebrow="Modality-wise masking"
          title="Vulnerability matrix"
          sub="Each cell: one modality suppressed at one severity, averaged over held-out subjects."
          actions={
            <div className="flex flex-wrap items-center gap-2">
              <Segmented size="sm" value={pipeline} onChange={setPipeline} options={[{ value: "fixed", label: "Fixed" }, { value: "def", label: "Defended" }]} />
              <Segmented size="sm" value={fill} onChange={setFill} options={fills.map((f) => ({ value: f, label: f }))} />
              <Segmented size="sm" value={metric} onChange={setMetric} options={[{ value: "drop", label: "ΔF1" }, { value: "asr", label: "ASR" }, { value: "cwrong", label: "Conf. wrong" }]} />
            </div>
          }
        />
        <div className="p-3">
          <EChart option={heat} height={60 + mods.length * 46} />
        </div>
      </Card>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {mods.map((m) => (
          <SweepCard key={m} m={m} fusion={fusion} clean={clean} family="mask" fill={fill} title={`${MODALITY_LABEL[m]} · ${fill}-mask`} />
        ))}
        {mods.map((m) => (
          <SweepCard key={`a${m}`} m={m} fusion={fusion} clean={clean} family="attenuate" title={`${MODALITY_LABEL[m]} · attenuation`} />
        ))}
      </div>
      <Card>
        <CardHeader eyebrow="H2" title="Suppression Vulnerability Index" sub="Normalised area under the fixed-fusion degradation–severity curve (zero-fill masking). Higher = more damaging to remove." />
        <div className="grid gap-3 p-5 md:grid-cols-4">
          {s.hypotheses.H2.ranking.map((r, i) => (
            <div key={r.modality} className="rounded-lg border border-line p-3">
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5 text-[13px] font-semibold">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ background: MODALITY_COLOR[r.modality] }} />
                  {MODALITY_LABEL[r.modality]}
                </span>
                <span className="num text-[11px] text-faint">#{i + 1}</span>
              </div>
              <div className="num mt-2 text-[20px] font-semibold">{r.svi?.toFixed(3)}</div>
              <div className="text-[11.5px] text-muted">mean full-mask drop {signed(-r.mean_drop)}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function SweepCard({ m, fusion, clean, family, fill, title }: { m: string; fusion: ConditionAgg[]; clean?: ConditionAgg; family: string; fill?: string; title: string }) {
  const option = useMemo(() => {
    const rows = fusion.filter((c) => c.family === family && c.modality === m && (fill == null || c.fill === fill)).sort((a, b) => (a.severity ?? 0) - (b.severity ?? 0));
    const xs = [0, ...rows.map((r) => r.severity as number)];
    const pts = (k: string) => [mm(clean, k), ...rows.map((r) => mm(r, k))];
    // CI band drawn as one closed polygon (lo forward, hi backward)
    const band = (k: string, color: string) => {
      const p = pts(k);
      const ok = xs.map((_, i) => p[i].lo != null && p[i].hi != null);
      const lo = xs.map((x, i) => [x * 100, p[i].lo as number]).filter((_, i) => ok[i]);
      const hi = xs.map((x, i) => [x * 100, p[i].hi as number]).filter((_, i) => ok[i]).reverse();
      return [
        {
          type: "custom",
          silent: true,
          z: 1,
          data: [0],
          renderItem: (_: any, api: any) => ({
            type: "polygon",
            shape: { points: [...lo, ...hi].map((pt) => api.coord(pt)) },
            style: { fill: color, opacity: 0.13 },
          }),
        },
      ];
    };
    const line = (k: string, name: string, color: string) => ({ type: "line", name, data: xs.map((x, i) => [x * 100, pts(k)[i].mean]), lineStyle: { color, width: 2 }, itemStyle: { color }, symbolSize: 6 });
    return {
      grid: { left: 6, right: 12, top: 12, bottom: 22 },
      tooltip: { trigger: "axis", formatter: (ps: any[]) => ttTitle(`${ps[0].value[0]}%`) + ps.filter((p) => p.seriesName).map((p) => ttRow(p.color, p.seriesName, (+p.value[1]).toFixed(3))).join("") },
      xAxis: { type: "value", min: 0, max: 100, axisLabel: { formatter: "{value}%" } },
      yAxis: { type: "value", min: 0, max: 1, splitNumber: 4 },
      series: [...band("fixed_f1", COLORS.fixed), ...band("def_f1", COLORS.defended), line("fixed_f1", "Fixed", COLORS.fixed), line("def_f1", "Defended", COLORS.defended)],
    };
  }, [m, fusion, clean, family, fill]);
  return (
    <Card>
      <div className="flex items-center justify-between px-4 pt-3 text-[12.5px] font-semibold">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full" style={{ background: MODALITY_COLOR[m] }} />
          {title}
        </span>
        <span className="flex gap-2 text-[10.5px] font-normal text-muted">
          <span className="flex items-center gap-1"><span className="h-0.5 w-3 bg-fixed" />fixed</span>
          <span className="flex items-center gap-1"><span className="h-0.5 w-3 bg-defended" />defended</span>
        </span>
      </div>
      <div className="px-2 pb-2">
        <EChart option={option} height={170} />
      </div>
    </Card>
  );
}

/* ------------------------------------------------------------------ defense */

function DefenseTab({ s, fusion }: { s: Summary; fusion: ConditionAgg[] }) {
  const H3 = s.hypotheses.H3;
  const forest = useMemo(() => {
    const subjects = Array.from(new Set(s.forest.map((r) => r.subject))).sort((a, b) => +a.slice(1) - +b.slice(1));
    return {
      grid: { left: 8, right: 16, top: 30, bottom: 26 },
      legend: { top: 0, left: 0, icon: "circle", itemWidth: 8, itemHeight: 8 },
      tooltip: { trigger: "item", formatter: (p: any) => ttTitle(`${p.value[1]} · ${p.seriesName}`) + ttRow(MODALITY_COLOR[p.data.m], "gain", signed(p.value[0])) + ttRow("#fff", "fixed → defended", `${p.data.f.toFixed(3)} → ${p.data.d.toFixed(3)}`) },
      xAxis: { type: "value", name: "defense gain (macro-F1, defended − fixed)", nameLocation: "middle", nameGap: 22 },
      yAxis: { type: "category", data: subjects, inverse: true },
      series: s.modalities.map((m, k) => ({
        type: "scatter",
        name: MODALITY_LABEL[m],
        data: s.forest.filter((r) => r.modality === m).map((r) => ({ value: [r.gain, r.subject], m, f: r.fixed_f1, d: r.def_f1 })),
        symbolSize: 9,
        symbolOffset: [0, (k - (s.modalities.length - 1) / 2) * 5],
        itemStyle: { color: MODALITY_COLOR[m], borderColor: "#fff", borderWidth: 1 },
        markLine: k === 0 ? { silent: true, symbol: "none", lineStyle: { color: COLORS.ink, type: "solid", width: 1 }, label: { show: false }, data: [{ xAxis: 0 }] } : undefined,
      })),
    };
  }, [s]);
  const conf = useMemo(() => {
    const fams = s.modalities.map((m) => fusion.find((c) => c.family === "mask" && c.modality === m && c.fill === "zero" && c.severity === 1));
    const clean = fusion.find((c) => c.family === "clean");
    const cats = ["clean", ...s.modalities.map((m) => MODALITY_LABEL[m])];
    const vals = (k: string) => [mm(clean, k).mean, ...fams.map((c) => mm(c, k).mean)];
    return {
      grid: { left: 8, right: 12, top: 56, bottom: 22 },
      legend: { top: 0, left: 0 },
      tooltip: { trigger: "axis", valueFormatter: (v: number) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`) },
      xAxis: { type: "category", data: cats },
      yAxis: { type: "value", max: 1, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` } },
      series: [
        { type: "bar", name: "Fixed: mean confidence", data: vals("fixed_conf"), itemStyle: { color: COLORS.fixed, opacity: 0.35, borderRadius: [3, 3, 0, 0] }, barGap: "8%" },
        { type: "bar", name: "Fixed: confidently wrong", data: vals("fixed_cwrong"), itemStyle: { color: COLORS.fixed, borderRadius: [3, 3, 0, 0] } },
        { type: "bar", name: "Defended: confidently wrong", data: vals("def_cwrong"), itemStyle: { color: COLORS.defended, borderRadius: [3, 3, 0, 0] } },
      ],
    };
  }, [s, fusion]);
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader eyebrow="H3" title="Mitigation under full single-modality suppression" sub={`Primary condition: ${s.hypotheses.primary_condition}. Clean cost of the defense: ${signed(-(H3.clean_cost.mean ?? 0))} macro-F1 (non-inferiority margin ${H3.margin}).`} />
        <table className="w-full text-[12.5px]">
          <thead>
            <tr className="text-left text-[11px] uppercase tracking-wide text-muted">
              <th className="px-5 py-2 font-medium">Suppressed</th>
              <th className="px-3 py-2 font-medium">Fixed macro-F1</th>
              <th className="px-3 py-2 font-medium">Defended macro-F1</th>
              <th className="px-3 py-2 font-medium">Gain</th>
              <th className="px-3 py-2 font-medium">p (Holm)</th>
              <th className="px-3 py-2 font-medium">Effect r</th>
              <th className="px-5 py-2 font-medium" />
            </tr>
          </thead>
          <tbody>
            {H3.per_modality.map((r: any) => (
              <tr key={r.modality} className="border-t border-line">
                <td className="px-5 py-2 font-medium">
                  <span className="flex items-center gap-1.5">
                    <span className="h-2 w-2 rounded-full" style={{ background: MODALITY_COLOR[r.modality] }} />
                    {MODALITY_LABEL[r.modality]}
                  </span>
                </td>
                <td className="px-3 py-2"><CiText ci={r.fixed_f1} /></td>
                <td className="px-3 py-2"><CiText ci={r.def_f1} /></td>
                <td className="px-3 py-2"><CiText ci={r.gain} /></td>
                <td className="num px-3 py-2">{pval(r.p_holm)}</td>
                <td className="num px-3 py-2">{r.rank_biserial?.toFixed(2) ?? "—"}</td>
                <td className="px-5 py-2 text-right">{r.significant ? <Badge tone="good">significant</Badge> : <Badge>n.s.</Badge>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      <div className="grid grid-cols-12 gap-4">
        <Card className="col-span-12 xl:col-span-7">
          <CardHeader eyebrow="Per subject" title="Defense gain by held-out subject" sub="Full zero-fill masking of each modality. Points right of the line are subjects the defense helped." />
          <div className="p-3">
            <EChart option={forest} height={Math.max(260, 30 + new Set(s.forest.map((r) => r.subject)).size * 22)} />
          </div>
        </Card>
        <Card className="col-span-12 xl:col-span-5">
          <CardHeader eyebrow="Confidence distortion" title="Suppression makes fixed fusion confidently wrong" sub="Share of windows that are wrong with confidence ≥ 0.8." />
          <div className="p-3">
            <EChart option={conf} height={260} />
          </div>
        </Card>
      </div>
      {s.moddrop_comparison && (
        <Card>
          <CardHeader eyebrow="Extension" title="Is robustness training enough?" sub="ModDrop trains the same architecture with random modality dropout (generic missingness). Compared under the same attacks." />
          <table className="w-full text-[12.5px]">
            <thead>
              <tr className="text-left text-[11px] uppercase tracking-wide text-muted">
                <th className="px-5 py-2 font-medium">Attack family</th>
                <th className="px-3 py-2 font-medium">Fusion · fixed</th>
                <th className="px-3 py-2 font-medium">Fusion · defended</th>
                <th className="px-3 py-2 font-medium">ModDrop · fixed</th>
                <th className="px-3 py-2 font-medium">ModDrop · defended</th>
              </tr>
            </thead>
            <tbody>
              {s.moddrop_comparison.map((r) => (
                <tr key={r.family} className="border-t border-line">
                  <td className="px-5 py-2 font-medium capitalize">{r.family.replace("_", " ")}</td>
                  <td className="px-3 py-2"><CiText ci={r.fusion_fixed} /></td>
                  <td className="px-3 py-2"><CiText ci={r.fusion_defended} /></td>
                  <td className="px-3 py-2"><CiText ci={r.moddrop_fixed} /></td>
                  <td className="px-3 py-2"><CiText ci={r.moddrop_defended} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ detection */

function DetectionTab({ s, det, fusion }: { s: Summary; det?: import("../lib/api").Detection; fusion: ConditionAgg[] }) {
  const [family, setFamily] = useState("mask");
  const H4 = s.hypotheses.H4;
  const roc = useMemo(() => {
    if (!det) return null;
    const pick = (m: string) =>
      Object.entries(det.curves).find(([k]) =>
        family === "mask" ? k.startsWith("mask|") && k.includes(`modality=${m}|`) && k.includes("fill=zero") : family === "hold" ? k.startsWith("mask|") && k.includes(`modality=${m}|`) && k.includes("fill=hold") : family === "attenuate" ? k.startsWith("attenuate|") && k.includes(`modality=${m}|`) && k.endsWith("severity=0.8") : k.startsWith("noise|") && k.includes(`modality=${m}|`),
      );
    const series: any[] = [{ type: "line", data: [[0, 0], [1, 1]], lineStyle: { color: COLORS.axis, type: "dashed", width: 1 }, symbol: "none", silent: true }];
    s.modalities.forEach((m) => {
      const c = pick(m);
      if (c) series.push({ type: "line", name: `${MODALITY_LABEL[m]} · AUROC ${c[1].auroc.toFixed(3)}`, data: c[1].fpr.map((x, i) => [x, c[1].tpr[i]]), symbol: "none", lineStyle: { color: MODALITY_COLOR[m], width: 2 }, itemStyle: { color: MODALITY_COLOR[m] } });
    });
    return {
      grid: { left: 36, right: 16, top: 34, bottom: 30 },
      legend: { top: 0, left: 0, itemWidth: 12 },
      tooltip: { trigger: "axis", formatter: (ps: any[]) => ps.filter((p) => p.seriesName).map((p) => ttRow(p.color, p.seriesName.split(" ·")[0], `TPR ${(+p.value[1]).toFixed(2)} @ FRR ${(+p.value[0]).toFixed(2)}`)).join("") },
      xAxis: { type: "value", min: 0, max: 1, name: "false-rejection rate (clean windows)", nameLocation: "middle", nameGap: 22 },
      yAxis: { type: "value", min: 0, max: 1, name: "detection rate", nameLocation: "middle", nameGap: 30 },
      series,
    };
  }, [det, family, s.modalities]);
  const matrix = useMemo(() => {
    if (!det) return null;
    const fams = ["mask", "attenuate", "noise", "random_drop", "targeted", "oracle", "adaptive"];
    const rows: { label: string; e: any }[] = [];
    fams.forEach((f) => {
      const entries = det.matrix.filter((e) => e.family === f);
      if (f === "mask" || f === "attenuate" || f === "noise") s.modalities.forEach((m) => { const e = entries.find((x) => x.modality === m); if (e) rows.push({ label: `${f} · ${MODALITY_LABEL[m]}`, e }); });
      else if (entries[0]) rows.push({ label: f.replace("_", " "), e: entries[0] });
    });
    const cols = [["auroc_Q", "Q"], ["auroc_D", "D"], ["auroc_U", "U"], ["auroc", "R (combined)"]];
    const data: [number, number, number][] = [];
    rows.forEach((r, i) => cols.forEach(([k], j) => data.push([j, i, r.e[k] ?? 0.5])));
    return {
      grid: { left: 8, right: 70, top: 8, bottom: 26 },
      tooltip: { formatter: (p: any) => ttTitle(rows[p.value[1]].label) + ttRow(BLUE_RAMP[4], `AUROC · ${cols[p.value[0]][1]}`, (+p.value[2]).toFixed(3)) },
      xAxis: { type: "category", data: cols.map((c) => c[1]), position: "top", axisLabel: { color: COLORS.ink2 } },
      yAxis: { type: "category", data: rows.map((r) => r.label), inverse: true, axisLabel: { color: COLORS.ink2 } },
      visualMap: { min: 0.5, max: 1, right: 0, top: "middle", itemHeight: 140, inRange: { color: ["#f2f4f7", ...BLUE_RAMP] }, textStyle: { fontSize: 10.5, color: COLORS.muted }, formatter: (v: number) => v.toFixed(2) },
      series: [{ type: "heatmap", data: data.map((d) => ({ value: d, label: { color: d[2] > 0.85 ? "#fff" : COLORS.ink } })), label: { show: true, fontSize: 10.5, fontFamily: MONO_FONT, formatter: (p: any) => fmt2(+p.value[2]) }, itemStyle: { borderColor: "#fff", borderWidth: 2, borderRadius: 3 } }],
    };
  }, [det, s.modalities]);
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="card-pad">
          <div className="eyebrow mb-1">Clean false-rejection rate</div>
          <div className="text-[22px] font-semibold"><CiText ci={H4.clean_frr} pct /></div>
          <div className="mt-1 text-xs text-muted">target α = {det?.alpha} by construction on exchangeable data; pre-registered limit {(H4.frr_max * 100).toFixed(0)}% on unseen subjects</div>
        </Card>
        <Card className="card-pad md:col-span-2">
          <div className="eyebrow mb-2">H4 · per-modality AUROC (full masking)</div>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {H4.per_modality.map((r: any) => (
              <div key={r.modality}>
                <div className="flex items-center gap-1.5 text-[12.5px] font-medium">
                  <span className="h-2 w-2 rounded-full" style={{ background: MODALITY_COLOR[r.modality] }} />
                  {MODALITY_LABEL[r.modality]}
                </div>
                <div className="num text-[18px] font-semibold">{r.auroc.mean?.toFixed(3)}</div>
                <div className="text-[11px] text-muted">detected {((r.tpr.mean ?? 0) * 100).toFixed(0)}% · p {pval(r.p_holm)}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-12 gap-4">
        <Card className="col-span-12 xl:col-span-6">
          <CardHeader
            eyebrow="Pooled over subjects"
            title="Detection ROC"
            actions={<Segmented size="sm" value={family} onChange={setFamily} options={[{ value: "mask", label: "Zero mask" }, { value: "hold", label: "Hold" }, { value: "attenuate", label: "Atten. 80%" }, { value: "noise", label: "Noise 5 dB" }]} />}
          />
          <div className="p-3">{roc ? <EChart option={roc} height={330} /> : <Skeleton className="h-80" />}</div>
        </Card>
        <Card className="col-span-12 xl:col-span-6">
          <CardHeader eyebrow="Signature analysis" title="Detectability matrix" sub="Which evidence channel exposes which attack (AUROC, attacked vs. clean windows)." actions={<InfoTip>Rows: attack and target. Columns: each evidence channel alone (max over modalities) and the combined conformal score R.</InfoTip>} />
          <div className="p-3">{matrix ? <EChart option={matrix} height={Math.max(330, (det?.matrix.length ?? 10) * 19 + 40)} /> : <Skeleton className="h-80" />}</div>
        </Card>
      </div>
      <TprTable s={s} fusion={fusion} />
    </div>
  );
}

function TprTable({ s, fusion }: { s: Summary; fusion: ConditionAgg[] }) {
  const rows = s.modalities.flatMap((m) =>
    ["zero", "missing", "hold"].map((f) => ({ m, f, c: fusion.find((c) => c.family === "mask" && c.modality === m && c.fill === f && c.severity === 1) })),
  );
  return (
    <Card>
      <CardHeader eyebrow="Fill matters" title="How the suppression is filled changes its detectability" sub="Full masking with zeros, dropped samples or a frozen value." />
      <table className="w-full text-[12.5px]">
        <thead>
          <tr className="text-left text-[11px] uppercase tracking-wide text-muted">
            <th className="px-5 py-2 font-medium">Target</th>
            <th className="px-3 py-2 font-medium">Fill</th>
            <th className="px-3 py-2 font-medium">Fixed F1</th>
            <th className="px-3 py-2 font-medium">Defended F1</th>
            <th className="px-3 py-2 font-medium">Flagged</th>
            <th className="px-3 py-2 font-medium">AUROC</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ m, f, c }) => (
            <tr key={m + f} className="border-t border-line">
              <td className="px-5 py-1.5 font-medium">
                <span className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full" style={{ background: MODALITY_COLOR[m] }} />
                  {MODALITY_LABEL[m]}
                </span>
              </td>
              <td className="px-3 py-1.5 text-muted">{f}</td>
              <td className="px-3 py-1.5"><CiText ci={mm(c, "fixed_f1")} /></td>
              <td className="px-3 py-1.5"><CiText ci={mm(c, "def_f1")} /></td>
              <td className="px-3 py-1.5"><CiText ci={mm(c, "flag_rate")} pct /></td>
              <td className="px-3 py-1.5"><CiText ci={mm(c, "auroc")} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

/* ------------------------------------------------------------------ attackers */

function AttackersTab({ s }: { s: Summary }) {
  const att = s.hypotheses.secondary.attackers;
  const NAMES: Record<string, string> = { random_drop: "Random", targeted: "Targeted", oracle: "Oracle", adaptive: "Adaptive (defense-aware)" };
  const option = useMemo(
    () => ({
      grid: { left: 8, right: 12, top: 34, bottom: 22 },
      legend: { top: 0, left: 0 },
      tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: att.map((a) => NAMES[a.attacker] ?? a.attacker) },
      yAxis: { type: "value", min: 0, max: 1, name: "macro-F1 under attack" },
      series: [
        { type: "bar", name: "Fixed fusion", data: att.map((a) => a.fixed_f1.mean), itemStyle: { color: COLORS.fixed, borderRadius: [4, 4, 0, 0] }, barGap: "10%", barMaxWidth: 46 },
        { type: "bar", name: "Reliability-aware", data: att.map((a) => a.def_f1.mean), itemStyle: { color: COLORS.defended, borderRadius: [4, 4, 0, 0] }, barMaxWidth: 46 },
      ],
    }),
    [att],
  );
  return (
    <div className="grid grid-cols-12 gap-4">
      <Card className="col-span-12 xl:col-span-7">
        <CardHeader eyebrow="Threat model" title="Attacker capability ladder" sub="Every attacker suppresses exactly one modality per window (full masking), choosing it differently." />
        <div className="p-3">
          <EChart option={option} height={300} />
        </div>
      </Card>
      <Card className="col-span-12 xl:col-span-5">
        <CardHeader eyebrow="Attackers" title="What each attacker knows" />
        <div className="divide-y divide-line text-[12.5px]">
          {att.map((a) => (
            <div key={a.attacker} className="px-5 py-3">
              <div className="flex items-center justify-between">
                <span className="font-semibold">{NAMES[a.attacker] ?? a.attacker}</span>
                <span className="text-muted">
                  flagged <span className="num text-ink">{((a.flag_rate.mean ?? 0) * 100).toFixed(0)}%</span>
                </span>
              </div>
              <p className="mt-0.5 text-muted">
                {a.attacker === "random_drop" && "Benign-like: a random modality disappears (what missing-modality robustness is usually tested against)."}
                {a.attacker === "targeted" && "Label-free and query-based: suppresses the modality whose removal moves the fixed-fusion output the most."}
                {a.attacker === "oracle" && "Knows the true label: picks the modality that minimises the probability of the correct class (worst case)."}
                {a.attacker === "adaptive" && "Knows the defense: searches 160 candidates for the cheapest suppression that flips the defended decision without raising the flag."}
              </p>
              <div className="mt-1.5 flex gap-4 text-[11.5px]">
                <span>
                  ASR fixed <span className="num text-ink">{((a.fixed_asr.mean ?? 0) * 100).toFixed(1)}%</span>
                </span>
                <span>
                  ASR defended <span className="num text-ink">{((a.def_asr.mean ?? 0) * 100).toFixed(1)}%</span>
                </span>
                <span>
                  evasive flips <span className="num text-ink">{((a.evasive_def_flip.mean ?? 0) * 100).toFixed(1)}%</span>
                </span>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------ bursts */

function BurstsTab({ s, fusion }: { s: Summary; fusion: ConditionAgg[] }) {
  const [scope, setScope] = useState("full");
  const rows = fusion.filter((c) => c.family === "burst" && c.scope === scope);
  const Bs = Array.from(new Set(rows.map((r) => r.burst as number))).sort((a, b) => a - b);
  const mk = (k: string, transform: (v: number | null) => number | null, title: string) => ({
    grid: { left: 8, right: 12, top: 30, bottom: 26 },
    legend: { top: 0, left: 0 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: Bs.map(String), name: "burst length (windows)", nameLocation: "middle", nameGap: 22 },
    yAxis: { type: "value", name: title, min: 0 },
    series: s.modalities.flatMap((m) => [
      ...(k === "err"
        ? [
            { type: "line", name: `${MODALITY_LABEL[m]} fixed`, data: Bs.map((b) => transform(mm(rows.find((r) => r.modality === m && r.burst === b), "fixed_acc").mean)), lineStyle: { color: MODALITY_COLOR[m], type: "dashed", width: 1.6 }, itemStyle: { color: MODALITY_COLOR[m] }, symbolSize: 5 },
            { type: "line", name: `${MODALITY_LABEL[m]} defended`, data: Bs.map((b) => transform(mm(rows.find((r) => r.modality === m && r.burst === b), "def_acc").mean)), lineStyle: { color: MODALITY_COLOR[m], width: 2 }, itemStyle: { color: MODALITY_COLOR[m] }, symbolSize: 6 },
          ]
        : [{ type: "line", name: MODALITY_LABEL[m], data: Bs.map((b) => transform(mm(rows.find((r) => r.modality === m && r.burst === b), k).mean)), lineStyle: { color: MODALITY_COLOR[m], width: 2 }, itemStyle: { color: MODALITY_COLOR[m] }, symbolSize: 6 }]),
    ]),
  });
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Segmented value={scope} onChange={setScope} options={[{ value: "full", label: "Fully suppressed windows" }, { value: "partial", label: "Edge windows (partly overlapped)" }]} />
        <InfoTip>Bursts suppress the continuous stream for B consecutive windows. Overlapping windows at the burst edges see only part of it.</InfoTip>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card>
          <CardHeader eyebrow="Persistence" title="Error rate inside the burst" sub="Dashed = fixed fusion, solid = reliability-aware." />
          <div className="p-3"><EChart option={mk("err", (v) => (v == null ? null : 1 - v), "error rate")} height={300} /></div>
        </Card>
        <Card>
          <CardHeader eyebrow="Detection" title="Share of burst windows flagged" />
          <div className="p-3"><EChart option={mk("flag_rate", (v) => v, "flagged")} height={300} /></div>
        </Card>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ ablations */

function AblationsTab({ s }: { s: Summary; fusion: ConditionAgg[] }) {
  const points = s.operating_points ?? [];
  if (!points.length) return <Card><Empty title="No ablations in this run" /></Card>;
  const pct = (c?: CI) => (c?.mean == null ? "—" : `${(c.mean * 100).toFixed(1)}%`);
  const num = (c?: CI) => (c?.mean == null ? "—" : c.mean.toFixed(3));
  const frrMax = s.hypotheses.H4.frr_max;
  return (
    <Card>
      <CardHeader
        eyebrow="Ablation"
        title="Which evidence and which calibration reference do the work?"
        sub="The report's defense (highlighted) next to evidence subsets, a decision-level product-of-experts variant, and the pooled-reference extension. Attack columns average over target modalities under full suppression."
      />
      <table className="w-full text-[12.5px]">
        <thead>
          <tr className="text-left text-[11px] uppercase tracking-wide text-muted">
            <th className="px-5 py-2 font-medium">Evidence</th>
            <th className="px-3 py-2 font-medium">Calibration</th>
            <th className="px-3 py-2 font-medium">Fusion</th>
            <th className="px-3 py-2 text-right font-medium">Clean false flags</th>
            <th className="px-3 py-2 text-right font-medium">Clean F1</th>
            <th className="px-3 py-2 text-right font-medium">Suppressed · fixed F1</th>
            <th className="px-3 py-2 text-right font-medium">Suppressed · defended F1</th>
            <th className="px-3 py-2 text-right font-medium">Suppressed · flagged</th>
          </tr>
        </thead>
        <tbody>
          {points.map((p) => (
            <tr key={`${p.components}|${p.strategy}|${p.calibration}`} className={clsx("border-t border-line", p.primary && "bg-defended-soft/50")}>
              <td className="px-5 py-1.5 font-mono font-medium">{p.components.split("").join(" + ")}</td>
              <td className="px-3 py-1.5 text-muted">{p.calibration === "pooled" ? "pooled (extension)" : "validation (report)"}</td>
              <td className="px-3 py-1.5 text-muted">{p.strategy === "poe" ? "product of experts" : "embedding gate"}</td>
              <td className={clsx("num px-3 py-1.5 text-right", p.clean_frr?.mean != null && p.clean_frr.mean > frrMax && "font-medium text-flag-ink")}>{pct(p.clean_frr)}</td>
              <td className="num px-3 py-1.5 text-right">{num(p.clean_def_f1)}</td>
              <td className="num px-3 py-1.5 text-right text-muted">{num(p.full_fixed_f1)}</td>
              <td className="num px-3 py-1.5 text-right">{num(p.full_def_f1)}</td>
              <td className="num px-3 py-1.5 text-right">{pct(p.full_flag_rate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="px-5 py-3 text-[11.5px] text-muted">
        The highlighted row is the configuration used everywhere else: the defense as specified in the project report. False-flag rates above the pre-registered limit
        ({(frrMax * 100).toFixed(0)}%) are marked. {s.n_subjects} subjects.
      </p>
    </Card>
  );
}

/* ------------------------------------------------------------------ manifest */

function ManifestTab({ runId, classes }: { runId: string; classes: string[] }) {
  const list = useQuery({ queryKey: ["manifest", runId], queryFn: () => api.manifestConditions(runId) });
  const [cond, setCond] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const current = cond ?? list.data?.conditions.find((c) => c.includes("modality=eda") && c.includes("severity=1")) ?? list.data?.conditions[0] ?? null;
  const rows = useQuery({
    queryKey: ["manifest-rows", runId, current, offset],
    queryFn: () => api.manifestRows(runId, current!, 100, offset),
    enabled: !!current,
    placeholderData: keepPreviousData,
  });
  if (list.isError) return <Card><Empty title="This run has no manifest" /></Card>;
  return (
    <Card>
      <CardHeader
        eyebrow="Traceability"
        title="Attacked-window manifest"
        sub="Every attacked window with its target, attack parameters, both decisions, reliability and flag (primary seed)."
        actions={
          <>
            <Select value={current ?? ""} onChange={(c) => { setCond(c); setOffset(0); }} options={(list.data?.conditions ?? []).map((c) => ({ value: c, label: c }))} className="w-[340px] font-mono text-[11.5px]" />
            {current && (
              <a href={downloadUrl.manifestCsv(runId, current)} className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-line-strong bg-surface px-3 text-[13px] font-medium hover:bg-sunken">
                <Download className="h-3.5 w-3.5" /> CSV
              </a>
            )}
          </>
        }
      />
      {rows.data ? (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-[12px]">
              <thead>
                <tr className="text-left text-[11px] uppercase tracking-wide text-muted">
                  {["subject", "t", "truth", "fixed", "defended", "fixed conf", "def conf", "R target", "w target", "flag"].map((h) => (
                    <th key={h} className="px-3 py-2 font-medium">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.data.rows.map((r, i) => (
                  <tr key={i} className="border-t border-line">
                    <td className="px-3 py-1">{r.subject}</td>
                    <td className="num px-3 py-1">{r.start_s}s</td>
                    <td className="px-3 py-1">{classes[r.y as number]}</td>
                    <td className={clsx("px-3 py-1", r.fixed_pred !== r.y && "font-medium text-attack")}>{classes[r.fixed_pred as number]}</td>
                    <td className={clsx("px-3 py-1", r.def_pred !== r.y && "font-medium text-attack")}>{classes[r.def_pred as number]}</td>
                    <td className="num px-3 py-1">{(+r.fixed_conf).toFixed(2)}</td>
                    <td className="num px-3 py-1">{(+r.def_conf).toFixed(2)}</td>
                    <td className="num px-3 py-1">{(+r.r_target).toFixed(3)}</td>
                    <td className="num px-3 py-1">{(+r.w_target).toFixed(2)}</td>
                    <td className="px-3 py-1">{r.flag ? <span className="text-flag-ink">⚑</span> : ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex items-center justify-between border-t border-line px-5 py-2.5 text-[12px] text-muted">
            <span>
              {offset + 1}–{Math.min(offset + 100, rows.data.total)} of {rows.data.total.toLocaleString()} windows
            </span>
            <div className="flex gap-2">
              <button className="link disabled:opacity-40" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 100))}>
                Previous
              </button>
              <button className="link disabled:opacity-40" disabled={offset + 100 >= rows.data.total} onClick={() => setOffset(offset + 100)}>
                Next
              </button>
            </div>
          </div>
        </>
      ) : (
        <Skeleton className="m-5 h-64" />
      )}
    </Card>
  );
}
