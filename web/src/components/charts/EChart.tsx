import * as echarts from "echarts/core";
import { BarChart, CustomChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  MarkPointComponent,
  TitleComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import ReactEChartsCore from "echarts-for-react/lib/core";
import type { CSSProperties } from "react";
import { COLORS } from "../../lib/palette";

echarts.use([
  LineChart, BarChart, ScatterChart, HeatmapChart, CustomChart,
  GridComponent, TooltipComponent, LegendComponent, MarkAreaComponent, MarkLineComponent, MarkPointComponent,
  DataZoomComponent, VisualMapComponent, TitleComponent, CanvasRenderer,
]);

const FONT = '"IBM Plex Sans", system-ui, sans-serif';
const MONO = '"IBM Plex Mono", ui-monospace, monospace';

echarts.registerTheme("trustfuse", {
  color: [COLORS.fixed, COLORS.defended, "#1baf7a", "#eda100", "#e87ba4", "#6250d6"],
  backgroundColor: "transparent",
  textStyle: { fontFamily: FONT, color: COLORS.ink2 },
  grid: { containLabel: true },
  categoryAxis: {
    axisLine: { lineStyle: { color: COLORS.axis } },
    axisTick: { show: false },
    axisLabel: { color: COLORS.muted, fontSize: 11 },
    splitLine: { show: false },
  },
  valueAxis: {
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: COLORS.muted, fontSize: 11, fontFamily: MONO },
    splitLine: { lineStyle: { color: COLORS.grid } },
    nameTextStyle: { color: COLORS.muted, fontSize: 11 },
  },
  line: { symbolSize: 7, lineStyle: { width: 2 }, emphasis: { lineStyle: { width: 2.5 } } },
  legend: { textStyle: { color: COLORS.ink2, fontSize: 12 }, itemWidth: 14, itemHeight: 3, icon: "roundRect" },
  tooltip: {
    backgroundColor: "#101828",
    borderWidth: 0,
    padding: [8, 10],
    textStyle: { color: "#fff", fontSize: 12, fontFamily: FONT },
    extraCssText: "border-radius:8px;box-shadow:0 12px 32px -8px rgba(16,24,40,.35);",
    axisPointer: { lineStyle: { color: COLORS.faint, type: "dashed" }, crossStyle: { color: COLORS.faint } },
  },
});

export const MONO_FONT = MONO;

export default function EChart({
  option,
  height = 260,
  style,
  onEvents,
  notMerge = true,
}: {
  option: echarts.EChartsCoreOption;
  height?: number | string;
  style?: CSSProperties;
  onEvents?: Record<string, (e: any) => void>;
  notMerge?: boolean;
}) {
  return (
    <ReactEChartsCore
      echarts={echarts}
      option={option}
      theme="trustfuse"
      notMerge={notMerge}
      lazyUpdate
      onEvents={onEvents}
      style={{ height, width: "100%", ...style }}
      opts={{ renderer: "canvas" }}
    />
  );
}

/** Tooltip row helper: coloured key mark + label + right-aligned value (text stays in ink tones). */
export function ttRow(color: string, label: string, value: string) {
  return `<div style="display:flex;align-items:center;gap:8px;min-width:170px"><span style="width:8px;height:8px;border-radius:2px;background:${color}"></span><span style="flex:1;opacity:.85">${label}</span><span style="font-family:${MONO}">${value}</span></div>`;
}

export function ttTitle(text: string) {
  return `<div style="font-weight:600;margin-bottom:4px">${text}</div>`;
}
