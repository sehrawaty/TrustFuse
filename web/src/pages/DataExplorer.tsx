import { keepPreviousData, useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { useEffect, useMemo, useRef, useState } from "react";
import EChart, { MONO_FONT } from "../components/charts/EChart";
import { Badge, Button, Card, CardHeader, ErrorNote, InfoTip, PageHeader, Select, Skeleton, Switch } from "../components/ui";
import { api, type Signals } from "../lib/api";
import { clock, seconds } from "../lib/format";
import { CLASS_INK, COLORS, MODALITY_COLOR, MODALITY_LABEL } from "../lib/palette";
import { useApp } from "../lib/store";

// Protocol-condition bands: muted, non-status hues (red/amber/green stay reserved for attack/flag/good).
const RAW_BAND: Record<string, string> = {
  baseline: "#e4e8ee", stress: "#cdd3dd", amusement: "#dde4f2", meditation: "#e8e1f2", transient: "#ffffff", ignore: "#ffffff",
  sitting: "#e4e8ee", stairs: "#cdd3dd", "table soccer": "#e8e1f2", cycling: "#dde9e3", driving: "#efe6d8", lunch: "#efe2e9", walking: "#dde4f2", working: "#e1e6ec",
};

export default function DataExplorer() {
  const { task, subject: subjMap, setSubject } = useApp();
  const ov = useQuery({ queryKey: ["overview"], queryFn: api.overview });
  const taskInfo = ov.data?.tasks.find((t) => t.key === task);
  const subjects = useQuery({ queryKey: ["subjects", task], queryFn: () => api.subjects(task) });
  const subject = subjMap[task] ?? "S2";
  const [range, setRange] = useState<[number, number] | null>(null);
  const [showWindows, setShowWindows] = useState(true);
  useEffect(() => setRange(null), [task, subject]);
  const full = useQuery({ queryKey: ["signals", task, subject, "full"], queryFn: () => api.signals(task, subject, undefined, undefined, 1200) });
  const zoom = useQuery({
    queryKey: ["signals", task, subject, range],
    queryFn: () => api.signals(task, subject, range![0], range![1], 2000),
    enabled: !!range && range[1] - range[0] < (full.data?.duration_s ?? 1e9) * 0.6,
    placeholderData: keepPreviousData,
  });
  const windows = useQuery({ queryKey: ["windows", task, subject], queryFn: () => api.windows(task, subject) });
  const data = (range && zoom.data) || full.data;
  const mods = taskInfo?.modalities ?? [];
  const sInfo = subjects.data?.find((s) => s.subject === subject);

  return (
    <div>
      <PageHeader
        eyebrow="Explore"
        title="Data Explorer"
        description="Synchronized wrist streams of every subject, exactly as the models consume them: resampled to model rates, in physical units, with the protocol labels underneath. Drag or scroll on a lane to zoom in; the view refetches at full resolution."
      />
      <div className="grid grid-cols-12 gap-4">
        <Card className="col-span-12 xl:col-span-9">
          <CardHeader
            eyebrow={taskInfo?.dataset_name}
            title={
              <span className="flex items-center gap-2">
                Subject
                <Select value={subject} onChange={(s) => setSubject(task, s)} options={(subjects.data ?? []).map((s) => ({ value: s.subject, label: s.subject }))} className="h-7 w-[92px]" />
                {data && (
                  <span className="num text-[12px] font-normal text-muted">
                    {clock(data.t0)} – {clock(data.t1)}
                  </span>
                )}
              </span>
            }
            actions={
              <>
                <Switch checked={showWindows} onChange={setShowWindows} label="Evaluation windows" />
                {range && (
                  <Button size="sm" variant="ghost" onClick={() => setRange(null)}>
                    Reset zoom
                  </Button>
                )}
              </>
            }
          />
          <div className={clsx("p-3 transition-opacity", zoom.isFetching && "opacity-80")}>
            {full.isError && <ErrorNote error={full.error} />}
            {!data ? <Skeleton className="h-[520px]" /> : <SessionChart data={data} mods={mods} windows={showWindows ? windows.data : undefined} onZoom={setRange} fullDuration={full.data?.duration_s ?? 0} />}
          </div>
        </Card>
        <div className="col-span-12 space-y-4 xl:col-span-3">
          <Card>
            <CardHeader eyebrow="This subject" title={subject} />
            <div className="space-y-3 p-4 text-[12.5px]">
              {sInfo ? (
                <>
                  <Row k="Recording" v={seconds(sInfo.duration_s)} />
                  <Row k="Evaluation windows" v={String(sInfo.windows)} />
                  <Row k="LOSO fold model" v={sInfo.has_model ? "available" : "missing"} />
                  <div>
                    <div className="eyebrow mb-1.5">Class balance (windows)</div>
                    <div className="flex h-2.5 overflow-hidden rounded-full ring-1 ring-inset ring-line">
                      {sInfo.class_windows.map((c, i) => (
                        <div key={i} style={{ width: `${(c / sInfo.windows) * 100}%`, background: CLASS_INK[i], opacity: 0.6 }} />
                      ))}
                    </div>
                    <div className="mt-1.5 space-y-0.5">
                      {taskInfo?.classes.map((c, i) => (
                        <div key={c} className="flex justify-between text-muted">
                          <span className="flex items-center gap-1.5">
                            <span className="h-2 w-2 rounded-sm" style={{ background: CLASS_INK[i], opacity: 0.6 }} />
                            {c}
                          </span>
                          <span className="num text-ink">{sInfo.class_windows[i]}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <Skeleton className="h-40" />
              )}
            </div>
          </Card>
          <Card>
            <CardHeader eyebrow="Preprocessing" title="Model inputs" />
            <div className="space-y-2 p-4 text-[12.5px]">
              <Row k="Window / stride" v={taskInfo ? `${taskInfo.window_s} s / ${taskInfo.stride_s} s` : "—"} />
              {mods.map((m) => (
                <div key={m} className="flex items-center justify-between">
                  <span className="flex items-center gap-1.5 text-muted">
                    <span className="h-2 w-2 rounded-full" style={{ background: MODALITY_COLOR[m] }} />
                    {MODALITY_LABEL[m]}
                  </span>
                  <span className="num text-ink">
                    {ov.data?.modalities[m]?.native_hz} → {taskInfo?.model_hz[m]} Hz
                  </span>
                </div>
              ))}
              <p className="border-t border-line pt-2 text-[11.5px] leading-relaxed text-muted">
                Windows are label-pure (every sample has the same class). z-normalisation statistics come from the training subjects of each fold only.
                <InfoTip>Leakage control: the held-out subject never contributes to normalisation, early stopping or reliability calibration.</InfoTip>
              </p>
            </div>
          </Card>
        </div>
        <Card className="col-span-12">
          <CardHeader eyebrow="All subjects" title="Cohort overview" sub="Window counts per class after label-purity filtering." />
          <div className="overflow-x-auto p-2">
            <table className="w-full text-[12.5px]">
              <thead>
                <tr className="text-left text-[11px] uppercase tracking-wide text-muted">
                  <th className="px-3 py-2 font-medium">Subject</th>
                  <th className="px-3 py-2 font-medium">Duration</th>
                  <th className="px-3 py-2 font-medium">Windows</th>
                  {taskInfo?.classes.map((c) => (
                    <th key={c} className="px-3 py-2 font-medium">
                      {c}
                    </th>
                  ))}
                  <th className="px-3 py-2 font-medium">Fold model</th>
                </tr>
              </thead>
              <tbody>
                {(subjects.data ?? []).map((s) => (
                  <tr key={s.subject} onClick={() => setSubject(task, s.subject)} className={clsx("cursor-pointer border-t border-line hover:bg-sunken", s.subject === subject && "bg-sunken")}>
                    <td className="px-3 py-1.5 font-medium">{s.subject}</td>
                    <td className="num px-3 py-1.5">{clock(s.duration_s)}</td>
                    <td className="num px-3 py-1.5">{s.windows}</td>
                    {s.class_windows.map((c, i) => (
                      <td key={i} className="num px-3 py-1.5 text-ink-2">
                        {c}
                      </td>
                    ))}
                    <td className="px-3 py-1.5">{s.has_model ? <Badge tone="good">trained</Badge> : <Badge>missing</Badge>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-muted">{k}</span>
      <span className="num text-ink">{v}</span>
    </div>
  );
}

function SessionChart({ data, mods, windows, onZoom, fullDuration }: { data: Signals; mods: string[]; windows?: { start_s: number[]; window_s: number }; onZoom: (r: [number, number] | null) => void; fullDuration: number }) {
  const option = useMemo(() => {
    const laneH = 92;
    const top0 = 44;
    const grids = [{ left: 64, right: 18, top: 6, height: 26 }, ...mods.map((_, i) => ({ left: 64, right: 18, top: top0 + i * (laneH + 14), height: laneH }))];
    const xAxes = grids.map((_, i) => ({
      type: "value",
      gridIndex: i,
      min: data.t0,
      max: data.t1,
      show: i === grids.length - 1,
      axisLabel: { formatter: (v: number) => clock(v), fontFamily: MONO_FONT, fontSize: 10.5, color: COLORS.muted },
      splitLine: { show: false },
    }));
    const yAxes = [
      { type: "value", gridIndex: 0, show: false, min: 0, max: 1 },
      ...mods.map((m, i) => ({
        type: "value",
        gridIndex: i + 1,
        scale: true,
        splitNumber: 2,
        name: `${MODALITY_LABEL[m]}  ${data.modalities[m]?.unit ?? ""}`,
        nameLocation: "middle",
        nameGap: 46,
        nameTextStyle: { color: COLORS.ink2, fontSize: 11, fontWeight: 500 },
        axisLabel: { fontSize: 10, formatter: (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1)) },
      })),
    ];
    const segs = data.segments.filter((s) => s.name !== "transient");
    const series: any[] = [
      {
        type: "custom",
        xAxisIndex: 0,
        yAxisIndex: 0,
        silent: false,
        data: segs.map((s) => [Math.max(s.start, data.t0), Math.min(s.end, data.t1), s.name, s.class]),
        renderItem: (_: any, api: any) => {
          const a = api.coord([api.value(0), 1]);
          const b = api.coord([api.value(1), 0]);
          const w = Math.max(b[0] - a[0], 1);
          const name = api.value(2) as string;
          return {
            type: "group",
            children: [
              { type: "rect", shape: { x: a[0], y: a[1], width: w, height: b[1] - a[1], r: 3 }, style: { fill: RAW_BAND[name] ?? "#f2f4f7", stroke: "#fff", lineWidth: 1 } },
              w > 60 ? { type: "text", style: { text: name, x: a[0] + 7, y: (a[1] + b[1]) / 2, verticalAlign: "middle", fill: COLORS.ink2, font: "500 11px IBM Plex Sans" } } : null,
            ].filter(Boolean),
          };
        },
        tooltip: { formatter: (p: any) => `${p.value[2]}<br/><span style="opacity:.7">${clock(p.value[0])} – ${clock(p.value[1])}</span>` },
      },
    ];
    if (windows) {
      const ws = windows.start_s.filter((s) => s + windows.window_s >= data.t0 && s <= data.t1);
      const step = Math.max(1, Math.ceil(ws.length / 400));
      series.push({
        type: "custom",
        xAxisIndex: 0,
        yAxisIndex: 0,
        silent: true,
        data: ws.filter((_, i) => i % step === 0).map((s) => [s]),
        renderItem: (_: any, api: any) => {
          const a = api.coord([api.value(0), 0]);
          return { type: "rect", shape: { x: a[0], y: a[1] - 3, width: 1, height: 3 }, style: { fill: COLORS.ink2, opacity: 0.35 } };
        },
      });
    }
    mods.forEach((m, i) => {
      const sig = data.modalities[m];
      if (!sig) return;
      sig.values.forEach((ch, c) =>
        series.push({
          type: "line",
          name: `${MODALITY_LABEL[m]} ${sig.channels[c]}`,
          xAxisIndex: i + 1,
          yAxisIndex: i + 1,
          data: ch.map((v, k) => [sig.t[k], v]),
          showSymbol: false,
          sampling: "lttb",
          lineStyle: { width: 1.1, color: MODALITY_COLOR[m], opacity: 1 - c * 0.28 },
          animation: false,
        }),
      );
    });
    return {
      animation: false,
      grid: grids,
      xAxis: xAxes,
      yAxis: yAxes,
      tooltip: { trigger: "axis", axisPointer: { link: [{ xAxisIndex: "all" }] }, formatter: (ps: any[]) => (ps.length ? `<span style="font-family:${MONO_FONT}">${clock(ps[0].value[0])}</span><br/>` + ps.filter((p) => p.seriesType === "line").map((p) => `${p.seriesName}: <b>${(+p.value[1]).toFixed(3)}</b>`).join("<br/>") : "") },
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      dataZoom: [{ type: "inside", xAxisIndex: grids.map((_, i) => i), filterMode: "none" }],
      series,
    };
  }, [data, mods, windows]);
  const height = 44 + mods.length * 106 + 26;
  const timer = useRef<number>();
  return (
    <EChart
      option={option}
      height={height}
      onEvents={{
        datazoom: (e: any) => {
          const z = e.batch?.[0] ?? e;
          if (z.start == null) return;
          const span = data.t1 - data.t0;
          const a = data.t0 + (z.start / 100) * span;
          const b = data.t0 + (z.end / 100) * span;
          window.clearTimeout(timer.current);
          // debounce: refetch at full resolution once the user stops zooming
          timer.current = window.setTimeout(() => onZoom(b - a > fullDuration * 0.95 ? null : [Math.max(0, a), b]), 350);
        },
      }}
    />
  );
}
