import { keepPreviousData, useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { Shuffle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import EChart, { MONO_FONT, ttRow, ttTitle } from "../components/charts/EChart";
import { Button, Card, CardHeader, ErrorNote, InfoTip, PageHeader, Segmented, Select, Skeleton, Slider, Stat, Switch } from "../components/ui";
import { api, type Fill, type Timeline } from "../lib/api";
import { clock } from "../lib/format";
import { CLASS_INK, COLORS, MODALITY_COLOR, MODALITY_LABEL } from "../lib/palette";
import { useApp } from "../lib/store";

export default function StreamSimulator() {
  const { task, subject: subjMap, setSubject } = useApp();
  const ov = useQuery({ queryKey: ["overview"], queryFn: api.overview });
  const taskInfo = ov.data?.tasks.find((t) => t.key === task);
  const mods = taskInfo?.modalities ?? [];
  const subjects = useQuery({ queryKey: ["subjects", task], queryFn: () => api.subjects(task) });
  const subject = subjMap[task] ?? "S2";
  const [enabled, setEnabled] = useState(true);
  const [modality, setModality] = useState("eda");
  const [length, setLength] = useState(10);
  const [fill, setFill] = useState<Fill>("zero");
  const [start, setStart] = useState<number | null>(null);
  useEffect(() => {
    if (mods.length && !mods.includes(modality)) setModality(mods[mods.length > 2 ? 2 : 0]);
  }, [mods, modality]);
  const windows = useQuery({ queryKey: ["windows", task, subject], queryFn: () => api.windows(task, subject) });
  const n = windows.data?.start_s.length ?? 0;
  const startIdx = start ?? (windows.data ? Math.max(0, windows.data.y.findIndex((c) => c === 1) + 4) : 0);
  const burst = enabled && n ? { modality, start_index: Math.min(startIdx, n - 1), length, fill } : null;
  const tl = useQuery({
    queryKey: ["timeline", task, subject, burst],
    queryFn: () => api.timeline({ task, subject, burst }),
    enabled: !!windows.data && mods.includes(modality),
    placeholderData: keepPreviousData,
  });
  useEffect(() => setStart(null), [task, subject]);
  const t = tl.data;
  const S = t?.summary;

  return (
    <div>
      <PageHeader
        eyebrow="Attack & defend"
        title="Stream Simulator"
        description="Replay a whole held-out session window by window. A burst attack suppresses one stream for a contiguous run of windows and then restores it. Click the timeline to place the burst."
      />
      <Card className="mb-4">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3 px-5 py-3.5">
          <div className="flex items-center gap-2">
            <span className="eyebrow">Subject</span>
            <Select value={subject} onChange={(s) => setSubject(task, s)} options={(subjects.data ?? []).map((s) => ({ value: s.subject, label: s.subject }))} className="w-[96px]" />
          </div>
          <Switch checked={enabled} onChange={setEnabled} label="Burst attack" />
          <div className={clsx("flex items-center gap-2", !enabled && "opacity-40")}>
            <span className="eyebrow">Target</span>
            <Segmented size="sm" value={modality} onChange={setModality} options={mods.map((m) => ({ value: m, label: MODALITY_LABEL[m] }))} />
          </div>
          <div className={clsx("flex items-center gap-2", !enabled && "opacity-40")}>
            <span className="eyebrow">Fill</span>
            <Segmented size="sm" value={fill} onChange={(f) => setFill(f as Fill)} options={[{ value: "zero", label: "Zero" }, { value: "missing", label: "Missing" }, { value: "hold", label: "Hold" }]} />
          </div>
          <div className={clsx("flex w-[330px] items-center gap-3", !enabled && "opacity-40")}>
            <span className="eyebrow whitespace-nowrap">Length</span>
            <Slider value={length} onChange={(v) => setLength(Math.round(v))} min={1} max={40} step={1} label="Burst length" />
            <span className="num w-28 whitespace-nowrap text-right text-[12.5px]">
              {length} win · {taskInfo ? Math.round(((length - 1) * taskInfo.stride_s + taskInfo.window_s) / 6) / 10 : "–"} min
            </span>
          </div>
          <Button size="sm" icon={<Shuffle className="h-3.5 w-3.5" />} disabled={!enabled || !n} onClick={() => setStart(Math.floor(Math.random() * Math.max(n - length, 1)))}>
            Random placement
          </Button>
        </div>
      </Card>

      <div className="mb-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Card className="card-pad">
          <Stat label="Session accuracy · fixed" tone="fixed" value={S ? `${(S.fixed_acc * 100).toFixed(1)}%` : "—"} sub={S ? `clean ${(S.clean_fixed_acc * 100).toFixed(1)}%` : ""} />
        </Card>
        <Card className="card-pad">
          <Stat label="Session accuracy · defended" tone="defended" value={S ? `${(S.defended_acc * 100).toFixed(1)}%` : "—"} sub={S ? `clean ${(S.clean_defended_acc * 100).toFixed(1)}%` : ""} />
        </Card>
        <Card className="card-pad">
          <Stat label="Errors inside the burst" tone="attack" value={S ? `${S.fixed_errors_affected} → ${S.defended_errors_affected}` : "—"} sub={S ? `fixed → defended, ${S.affected} affected windows` : ""} />
        </Card>
        <Card className="card-pad">
          <Stat label="Burst windows flagged" tone="good" value={S && S.affected ? `${S.flagged_affected} / ${S.affected}` : "—"} sub={S ? `${S.flagged_clean} false flags on the clean remainder` : ""} />
        </Card>
      </div>

      <Card>
        <CardHeader
          eyebrow="Session timeline"
          title={t?.burst ? `Burst on ${MODALITY_LABEL[t.burst.modality]} · ${clock(t.burst.start_s)}–${clock(t.burst.end_s)}` : "Clean session"}
          sub="Rows: ground truth, fixed fusion, reliability-aware fusion, per-modality reliability (log scale). Tinted cells are wrong decisions."
          actions={<InfoTip>Windows overlap (stride {taskInfo?.stride_s}s, length {taskInfo?.window_s}s), so windows at the burst edges are only partly suppressed. The models are stateless: once a window no longer overlaps the burst, its decision is exactly the clean one.</InfoTip>}
        />
        <div className={clsx("p-3 transition-opacity", tl.isFetching && "opacity-75")}>
          {tl.isError && <ErrorNote error={tl.error} />}
          {!t ? <Skeleton className="h-[460px]" /> : <TimelineChart t={t} mods={mods} onPick={(i) => { setEnabled(true); setStart(i); }} />}
        </div>
      </Card>
    </div>
  );
}

function TimelineChart({ t, mods, onPick }: { t: Timeline; mods: string[]; onPick: (i: number) => void }) {
  const option = useMemo(() => {
    const x = t.start_s.map((s) => s + t.window_s / 2);
    const rows = ["Ground truth", "Fixed fusion", "Reliability-aware"];
    const band = (pred: number[], tone: "truth" | "fixed" | "def", rowIdx: number) => ({
      type: "custom",
      xAxisIndex: 0,
      yAxisIndex: 0,
      data: pred.map((p, i) => [x[i], rowIdx, p, i]),
      renderItem: (_: any, api: any) => {
        const i = api.value(3);
        const a = api.coord([t.start_s[i], api.value(1)]);
        const b = api.coord([t.start_s[i] + t.stride_s, api.value(1)]);
        const h = api.size([0, 1])[1] * 0.62;
        const cls = api.value(2);
        const wrong = tone !== "truth" && cls !== t.y[i];
        return {
          type: "rect",
          shape: { x: a[0], y: a[1] - h / 2, width: Math.max(b[0] - a[0], 1), height: h },
          style: { fill: wrong ? "rgba(208,59,59,0.85)" : CLASS_INK[cls], opacity: wrong ? 1 : tone === "truth" ? 0.55 : 0.4 },
        };
      },
      tooltip: {
        formatter: (p: any) => {
          const i = p.value[3];
          return (
            ttTitle(`${clock(t.start_s[i])} · window ${i + 1}`) +
            ttRow(CLASS_INK[t.y[i]], "truth", t.classes[t.y[i]]) +
            ttRow(COLORS.fixed, "fixed", t.classes[t.fixed_pred[i]]) +
            ttRow(COLORS.defended, "defended", t.classes[t.defended_pred[i]]) +
            ttRow(COLORS.flag, "flag", t.flag[i] ? "raised" : "—")
          );
        },
      },
    });
    const burstArea = t.burst
      ? { silent: true, itemStyle: { color: "rgba(208,59,59,0.06)", borderColor: "rgba(208,59,59,0.45)", borderWidth: 1, borderType: "dashed" }, data: [[{ xAxis: t.burst.start_s }, { xAxis: t.burst.end_s }]] }
      : undefined;
    const thr = 0.05 / mods.length;
    const flagged = t.flag.map((f, i) => (f ? [x[i], thr] : null)).filter(Boolean);
    return {
      animation: false,
      grid: [
        { left: 120, right: 20, top: 10, height: 96 },
        { left: 120, right: 20, top: 128, height: 210 },
      ],
      xAxis: [0, 1].map((g) => ({
        type: "value",
        gridIndex: g,
        min: t.start_s[0],
        max: t.start_s[t.start_s.length - 1] + t.window_s,
        show: g === 1,
        axisLabel: { formatter: (v: number) => clock(v), fontFamily: MONO_FONT, fontSize: 10.5 },
        splitLine: { show: false },
      })),
      yAxis: [
        { type: "category", gridIndex: 0, data: rows, inverse: true, axisLabel: { color: COLORS.ink2, fontSize: 11.5 }, axisLine: { show: false } },
        {
          type: "log",
          gridIndex: 1,
          min: 0.001,
          max: 1,
          name: "reliability R",
          nameLocation: "middle",
          nameGap: 64,
          nameTextStyle: { color: COLORS.ink2, fontSize: 11.5 },
          axisLabel: { formatter: (v: number) => (v >= 0.01 ? v.toString() : v.toExponential(0)) },
        },
      ],
      legend: { bottom: 0, data: mods.map((m) => MODALITY_LABEL[m]).concat(["flag"]), icon: "roundRect" },
      tooltip: { trigger: "item" },
      series: [
        band(t.y, "truth", 0),
        band(t.fixed_pred, "fixed", 1),
        { ...band(t.defended_pred, "def", 2), markArea: burstArea },
        ...mods.map((m, k) => ({
          type: "line",
          name: MODALITY_LABEL[m],
          xAxisIndex: 1,
          yAxisIndex: 1,
          // break the line where consecutive windows are not contiguous (excluded label segments)
          data: t.reliability[m].flatMap((r, i) => (i > 0 && t.start_s[i] - t.start_s[i - 1] > t.stride_s ? [[x[i] - t.stride_s, null], [x[i], Math.max(r, 0.001)]] : [[x[i], Math.max(r, 0.001)]])),
          connectNulls: false,
          showSymbol: false,
          lineStyle: { width: 1.6, color: MODALITY_COLOR[m] },
          itemStyle: { color: MODALITY_COLOR[m] },
          markArea: k === 0 ? burstArea : undefined,
          markLine:
            k === 0
              ? { silent: true, symbol: "none", lineStyle: { color: COLORS.ink, type: "dashed", width: 1 }, label: { formatter: "flag threshold α/M", color: COLORS.muted, fontSize: 10.5, position: "insideEndTop" }, data: [{ yAxis: thr }] }
              : undefined,
        })),
        {
          type: "scatter",
          name: "flag",
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: flagged,
          symbol: "triangle",
          symbolSize: 7,
          itemStyle: { color: COLORS.flag, borderColor: COLORS.flagInk, borderWidth: 0.6 },
          tooltip: { formatter: (p: any) => `flag raised · ${clock(p.value[0])}` },
        },
      ],
    };
  }, [t, mods]);
  return (
    <EChart
      option={option}
      height={380}
      onEvents={{
        click: (p: any) => {
          const xv = p.value?.[0];
          if (xv == null) return;
          let best = 0;
          let bd = Infinity;
          t.start_s.forEach((s, i) => {
            const d = Math.abs(s + t.window_s / 2 - xv);
            if (d < bd) {
              bd = d;
              best = i;
            }
          });
          onPick(best);
        },
      }}
    />
  );
}
