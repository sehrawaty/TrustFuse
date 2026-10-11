import { keepPreviousData, useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { useMemo } from "react";
import { Link } from "react-router-dom";
import EChart, { ttRow } from "../components/charts/EChart";
import Tex from "../components/Tex";
import { AttackBadge, Badge, Card, CardHeader, ErrorNote, FlagBadge, InfoTip, PageHeader, SectionTitle, Select, Skeleton } from "../components/ui";
import { api, type Calibration, type InferResult } from "../lib/api";
import { COLORS, MODALITY_COLOR, MODALITY_LABEL } from "../lib/palette";
import { defaultWindow, useApp, windowKey } from "../lib/store";

const FEATURE_TEXT: Record<string, string> = {
  nan_frac: "missing samples",
  zero_frac: "exact zeros",
  flat: "non-changing samples",
  max_run: "longest constant run",
  implaus: "physically implausible",
  level: "window level",
  log_std: "log amplitude",
  sqi: "cardiac-band power share",
};

export default function ReliabilityInspector() {
  const { task, subject: subjMap, setSubject, window: winMap, attack, components, toggleComponent, variant } = useApp();
  const ov = useQuery({ queryKey: ["overview"], queryFn: api.overview });
  const taskInfo = ov.data?.tasks.find((t) => t.key === task);
  const mods = taskInfo?.modalities ?? [];
  const subjects = useQuery({ queryKey: ["subjects", task], queryFn: () => api.subjects(task) });
  const subject = subjMap[task] ?? "S2";
  const windows = useQuery({ queryKey: ["windows", task, subject], queryFn: () => api.windows(task, subject) });
  const win = winMap[windowKey(task, subject)] ?? defaultWindow(windows.data);
  const comps = components.join("") || "Q";
  const calib = useQuery({ queryKey: ["calibration", task, subject, comps], queryFn: () => api.calibration(task, subject, comps), placeholderData: keepPreviousData });
  const infer = useQuery({
    queryKey: ["infer", task, subject, { attack, win, components: components.join(""), variant }],
    queryFn: () => api.infer({ task, subject, window: win, variant, attack, components }),
    placeholderData: keepPreviousData,
    enabled: !!windows.data,
  });
  const r = infer.data;
  const target = attack.type !== "none" ? attack.modality : null;

  return (
    <div>
      <PageHeader
        eyebrow="Attack & defend"
        title="Reliability Inspector"
        description={
          <>
            A transparent account of how the defense decides. Values are computed live for the window and attack currently loaded in the{" "}
            <Link to="/attack" className="link">
              Attack Lab
            </Link>
            . The calibration shown here was frozen on clean validation subjects before any attack was evaluated.
          </>
        }
        actions={
          <div className="flex items-center gap-2">
            <span className="eyebrow">Fold</span>
            <Select value={subject} onChange={(s) => setSubject(task, s)} options={(subjects.data ?? []).map((s) => ({ value: s.subject, label: s.subject }))} className="w-[96px]" />
          </div>
        }
      />

      <div className="grid grid-cols-12 gap-4">
        <Card className="col-span-12 xl:col-span-7">
          <CardHeader eyebrow="Method" title="From evidence to fusion weight" sub="Each step below applies independently to every modality m of the current window." />
          <div className="divide-y divide-line">
            <Eq n="1" title="Signal quality (model-free)" tex={String.raw`Q_m=\max_k\;\tilde z_{m,k},\qquad \tilde z_{m,k}=\delta_k\,\frac{f_{m,k}-\operatorname{med}_k}{\operatorname{IQR}_k/1.349}`}>
              Indicators f such as exact-zero share, longest constant run and plausibility are compared with all non-test subjects. The direction δ keeps only suspicious movement.
            </Eq>
            <Eq n="2" title="Cross-modal disagreement" tex={String.raw`D_m=\mathrm{JS}\!\left(g_m(h_m)\;\Big\|\;\tfrac{1}{M-1}\textstyle\sum_{j\neq m} g_j(h_j)\right)`}>
              The unimodal head of m is compared with the consensus of the other heads, in bits.
            </Eq>
            <Eq n="3" title="Predictive uncertainty" tex={String.raw`U_m=\frac{1}{\log K}\,\mathcal{H}\!\left[\tfrac1T\textstyle\sum_{t=1}^{T} g_m^{(t)}(h_m^{(t)})\right]`}>
              Entropy of the MC-dropout mean (T = 8 passes with reproducible masks).
            </Eq>
            <Eq n="4" title="Conformal reliability" tex={String.raw`a_m=Q_m+z(D_m)+z(U_m),\qquad R_m=\frac{1+\#\{i\in\text{val}: a^{(i)}_m\ge a_m\}}{n_{\text{val}}+1}`}>
              Under clean, exchangeable data P(R ≤ t) ≤ t. The flag fires when the smallest R falls to α/M or below.
            </Eq>
            <Eq n="5" title="Reliability-gated fusion" tex={String.raw`w_m=\operatorname{clip}\!\left(1-\frac{a_m-\tau_m}{\tau_m-\operatorname{med}(a_m)},0,1\right),\qquad \tilde h_m=w_m h_m+(1-w_m)\,\mu_m`}>
              τ is the conformal 1 − α/M threshold. The same frozen classifier receives the gated embeddings.
            </Eq>
          </div>
        </Card>

        <Card className="col-span-12 xl:col-span-5">
          <CardHeader
            eyebrow="Live values"
            title={r ? `Window ${r.window.index + 1} · ${r.window.label_name}` : "Loading…"}
            actions={r && (target ? <AttackBadge label={r.attack_label} /> : <Badge tone="good">clean</Badge>)}
          />
          {infer.isError && <div className="p-4"><ErrorNote error={infer.error} /></div>}
          {r ? (
            <div className="p-4">
              <table className="w-full text-[12.5px]">
                <thead>
                  <tr className="text-left text-[11.5px] text-muted">
                    <th className="pb-2 font-medium">m</th>
                    <th className="pb-2 text-right font-medium">Q</th>
                    <th className="pb-2 text-right font-medium">D</th>
                    <th className="pb-2 text-right font-medium">U</th>
                    <th className="pb-2 text-right font-medium">a</th>
                    <th className="pb-2 text-right font-medium">τ</th>
                    <th className="pb-2 text-right font-medium">R</th>
                    <th className="pb-2 text-right font-medium">w</th>
                  </tr>
                </thead>
                <tbody>
                  {mods.map((m) => {
                    const e = r.modalities[m];
                    return (
                      <tr key={m} className={clsx("border-t border-line", m === target && "bg-attack-soft/50")}>
                        <td className="py-2 font-medium">
                          <span className="flex items-center gap-1.5">
                            <span className="h-2 w-2 rounded-full" style={{ background: MODALITY_COLOR[m] }} />
                            {MODALITY_LABEL[m]}
                          </span>
                        </td>
                        <td className="num py-2 text-right">{e.Q.toFixed(2)}</td>
                        <td className="num py-2 text-right">{e.D.toFixed(3)}</td>
                        <td className="num py-2 text-right">{e.U.toFixed(3)}</td>
                        <td className="num py-2 text-right font-medium">{e.anomaly.toFixed(2)}</td>
                        <td className="num py-2 text-right text-muted">{e.tau.toFixed(2)}</td>
                        <td className={clsx("num py-2 text-right", e.R <= r.flag_threshold && "font-medium text-flag-ink")}>{e.R < 0.001 ? "<.001" : e.R.toFixed(3)}</td>
                        <td className="num py-2 text-right">{e.w.toFixed(2)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-3">
                <FlagBadge on={r.flag} />
                <span className="text-[12px] text-muted">
                  min R <span className="num text-ink">{r.min_r.toFixed(4)}</span> vs α/M <span className="num text-ink">{r.flag_threshold.toFixed(4)}</span>
                </span>
              </div>
              <div className="mt-4">
                <div className="eyebrow mb-2 flex items-center gap-1">
                  Evidence ablation <InfoTip>Disable an evidence channel. The conformal distribution is recomputed for the remaining sum, so the flag stays calibrated.</InfoTip>
                </div>
                <div className="flex gap-2">
                  {["Q", "D", "U"].map((c) => (
                    <button
                      key={c}
                      onClick={() => toggleComponent(c)}
                      className={clsx(
                        "flex-1 rounded-lg border py-1.5 text-[12.5px] font-medium",
                        components.includes(c) ? "border-defended/40 bg-defended-soft text-defended" : "border-line text-faint line-through",
                      )}
                    >
                      {c}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <Skeleton className="m-4 h-72" />
          )}
        </Card>
      </div>

      <SectionTitle aside={calib.data && <span className="text-[12px] text-muted">validation subjects {calib.data.model.split.val.join(", ")} · n = {Object.values(calib.data.modalities)[0]?.n_val} windows</span>}>
        Frozen calibration (clean validation)
      </SectionTitle>
      {calib.isError && <ErrorNote error={calib.error} />}
      <div className="grid gap-4 md:grid-cols-2">
        {calib.data
          ? mods.map((m) => <CalibCard key={m} m={m} c={calib.data!} live={r} />)
          : mods.map((m) => <Skeleton key={m} className="h-64" />)}
      </div>

      {r && (
        <>
          <SectionTitle>Signal-quality indicators of the current window</SectionTitle>
          <div className="grid gap-4 md:grid-cols-2">
            {mods.map((m) => {
              const q = r.modalities[m].quality;
              return (
                <Card key={m}>
                  <CardHeader title={<span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full" style={{ background: MODALITY_COLOR[m] }} />{MODALITY_LABEL[m]}{m === target && <span className="text-[11px] font-medium text-attack">attacked</span>}</span>} />
                  <table className="w-full text-[12.5px]">
                    <thead>
                      <tr className="text-left text-[11px] uppercase tracking-wide text-muted">
                        <th className="px-4 py-2 font-medium">Indicator</th>
                        <th className="px-4 py-2 text-right font-medium">Value</th>
                        <th className="px-4 py-2 text-right font-medium">Clean median</th>
                        <th className="px-4 py-2 text-right font-medium">Directional z</th>
                      </tr>
                    </thead>
                    <tbody>
                      {q.names.map((n, i) => (
                        <tr key={n} className="border-t border-line">
                          <td className="px-4 py-1.5">
                            <span className="font-mono text-[11.5px]">{n}</span> <span className="text-muted">· {FEATURE_TEXT[n] ?? ""}</span>
                          </td>
                          <td className="num px-4 py-1.5 text-right">{q.values[i].toFixed(3)}</td>
                          <td className="num px-4 py-1.5 text-right text-muted">{q.reference_median[i].toFixed(3)}</td>
                          <td className={clsx("num px-4 py-1.5 text-right", q.z[i] > 3 ? "font-medium text-flag-ink" : "text-ink-2")}>{q.z[i].toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Card>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

function Eq({ n, title, tex, children }: { n: string; title: string; tex: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[28px_1fr] gap-3 px-5 py-4">
      <span className="num mt-0.5 flex h-6 w-6 items-center justify-center rounded-full bg-sunken text-[11px] text-muted ring-1 ring-line">{n}</span>
      <div className="min-w-0">
        <div className="text-[13px] font-semibold">{title}</div>
        <div className="my-1 overflow-x-auto text-ink">
          <Tex block>{tex}</Tex>
        </div>
        <p className="text-[12.5px] leading-relaxed text-ink-2">{children}</p>
      </div>
    </div>
  );
}

function CalibCard({ m, c, live }: { m: string; c: Calibration; live?: InferResult }) {
  const cm = c.modalities[m];
  const option = useMemo(() => {
    const comps = [
      { k: "Q", label: "Q anomaly", color: COLORS.ink2 },
      { k: "D", label: "z(D)", color: "#7a8597" },
      { k: "U", label: "z(U)", color: "#b3bac6" },
    ];
    const series = comps.map((cp) => {
      const edges = cm.hist_edges[cp.k];
      const counts = cm.hist[cp.k];
      return {
        type: "bar",
        name: cp.label,
        barGap: "-100%",
        barWidth: 6,
        data: counts.map((v, i) => [(edges[i] + edges[i + 1]) / 2, v]),
        itemStyle: { color: cp.color, opacity: 0.55, borderRadius: [2, 2, 0, 0] },
      };
    });
    const liveA = live?.modalities[m]?.anomaly;
    return {
      grid: { left: 8, right: 16, top: 30, bottom: 26 },
      legend: { top: 0, right: 0, itemWidth: 10, itemHeight: 8 },
      tooltip: { trigger: "axis", formatter: (ps: any[]) => ps.map((p) => ttRow(p.color, p.seriesName, `${p.value[1]} windows`)).join("") },
      xAxis: { type: "value", name: "evidence (z units)", nameLocation: "middle", nameGap: 22, nameTextStyle: { fontSize: 10.5 } },
      yAxis: { type: "value", splitNumber: 3 },
      series: [
        ...series,
        {
          type: "line",
          data: [],
          markLine: {
            symbol: "none",
            silent: true,
            data: [
              { xAxis: cm.tau, lineStyle: { color: COLORS.ink, type: "dashed" }, label: { formatter: `τ = ${cm.tau.toFixed(1)}`, color: COLORS.ink2, fontSize: 10.5 } },
              ...(liveA != null ? [{ xAxis: Math.min(liveA, 60), lineStyle: { color: COLORS.attack, width: 2 }, label: { formatter: `this window a = ${liveA.toFixed(1)}`, color: COLORS.attack, fontSize: 10.5, position: "insideEndTop" } }] : []),
            ],
          },
        },
      ],
    };
  }, [cm, live, m]);
  return (
    <Card>
      <CardHeader
        title={
          <span className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full" style={{ background: MODALITY_COLOR[m] }} />
            {MODALITY_LABEL[m]}
          </span>
        }
        sub={`threshold τ ${cm.tau.toFixed(2)} · median a ${cm.median.toFixed(2)} · gate span ${cm.span.toFixed(2)}`}
      />
      <div className="px-3 pb-2 pt-1">
        <EChart option={option} height={200} />
      </div>
    </Card>
  );
}
