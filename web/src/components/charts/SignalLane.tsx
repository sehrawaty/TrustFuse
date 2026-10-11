import { useMemo } from "react";
import { COLORS, MODALITY_COLOR, MODALITY_LABEL } from "../../lib/palette";
import EChart, { MONO_FONT, ttRow } from "./EChart";

/** One modality lane: original signal (recessive grey) under the attacked signal (modality colour).
 *  Samples that differ are shaded as the suppressed region. */
export default function SignalLane({
  modality,
  hz,
  unit,
  original,
  attacked,
  channels,
  attacked_on,
  height = 92,
  t0 = 0,
}: {
  modality: string;
  hz: number;
  unit: string;
  original: number[][];
  attacked: (number | null)[][];
  channels: string[];
  attacked_on: boolean;
  height?: number;
  t0?: number;
}) {
  const option = useMemo(() => {
    const n = original[0].length;
    const t = Array.from({ length: n }, (_, i) => +(t0 + i / hz).toFixed(3));
    // contiguous ranges where the attacked signal differs from the original
    const diff: [number, number][] = [];
    if (attacked_on) {
      let start = -1;
      for (let i = 0; i < n; i++) {
        const d = original.some((ch, c) => {
          const a = attacked[c][i];
          return a == null || Math.abs(a - ch[i]) > 1e-6;
        });
        if (d && start < 0) start = i;
        if (!d && start >= 0) {
          diff.push([start, i - 1]);
          start = -1;
        }
      }
      if (start >= 0) diff.push([start, n - 1]);
    }
    const color = MODALITY_COLOR[modality];
    const series: any[] = [];
    original.forEach((ch, c) => {
      series.push({
        type: "line",
        name: `${channels[c]} original`,
        data: ch.map((v, i) => [t[i], v]),
        showSymbol: false,
        lineStyle: { width: 1.2, color: attacked_on ? "#c7ccd4" : color, opacity: attacked_on ? 1 : 1 - c * 0.25 },
        z: 1,
        silent: true,
        animation: false,
      });
    });
    if (attacked_on) {
      attacked.forEach((ch, c) => {
        series.push({
          type: "line",
          name: `${channels[c]} attacked`,
          data: ch.map((v, i) => [t[i], v]),
          showSymbol: false,
          connectNulls: false,
          lineStyle: { width: 1.5, color, opacity: 1 - c * 0.25 },
          z: 2,
          animation: false,
          markArea:
            c === 0 && diff.length
              ? {
                  silent: true,
                  itemStyle: { color: "rgba(208,59,59,0.07)", borderColor: "rgba(208,59,59,0.35)", borderWidth: 1, borderType: "dashed" },
                  data: diff.map(([a, b]) => [{ xAxis: t[a] }, { xAxis: t[b] }]),
                }
              : undefined,
        });
      });
    }
    return {
      animation: false,
      grid: { left: 6, right: 10, top: 8, bottom: 4 },
      xAxis: { type: "value", min: t[0], max: t[n - 1], show: false },
      yAxis: {
        type: "value",
        scale: true,
        splitNumber: 2,
        axisLabel: { fontSize: 10, color: COLORS.faint, fontFamily: MONO_FONT, formatter: (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1)) },
      },
      tooltip: {
        trigger: "axis",
        formatter: (ps: any[]) => {
          const p = ps.filter((x) => x.seriesName.endsWith(attacked_on ? "attacked" : "original"));
          return (
            `<div style="opacity:.7;margin-bottom:4px">${MODALITY_LABEL[modality]} · t = ${(+p[0]?.value[0]).toFixed(2)} s</div>` +
            p.map((x) => ttRow(color, x.seriesName.split(" ")[0], x.value[1] == null ? "missing" : `${(+x.value[1]).toFixed(3)} ${unit}`)).join("")
          );
        },
      },
      series,
    };
  }, [modality, hz, unit, original, attacked, channels, attacked_on, t0]);
  return <EChart option={option} height={height} />;
}
