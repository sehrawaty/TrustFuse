/** Colour roles shared by every chart (mirrors the CSS tokens and the paper figures). */
export const MODALITY_COLOR: Record<string, string> = {
  bvp: "#e87ba4",
  acc: "#6250d6",
  eda: "#1baf7a",
  temp: "#eda100",
};

export const MODALITY_LABEL: Record<string, string> = {
  bvp: "BVP",
  acc: "ACC",
  eda: "EDA",
  temp: "TEMP",
};

export const MODALITY_SHORT: Record<string, string> = {
  bvp: "PPG pulse",
  acc: "Motion",
  eda: "Skin conductance",
  temp: "Skin temp.",
};

export const MODALITY_LONG: Record<string, string> = {
  bvp: "Blood volume pulse (PPG)",
  acc: "3-axis accelerometer",
  eda: "Electrodermal activity",
  temp: "Skin temperature",
};

export const COLORS = {
  fixed: "#eb6834",
  defended: "#2a78d6",
  attack: "#d03b3b",
  flag: "#fab219",
  flagInk: "#935c06",
  good: "#0ca30c",
  goodInk: "#067647",
  ink: "#101828",
  ink2: "#344054",
  muted: "#667085",
  faint: "#98a2b3",
  grid: "#eaecf0",
  axis: "#d0d5dd",
  surface: "#ffffff",
  sunken: "#f9fafb",
};

/** Sequential blue ramp (magnitude), light -> dark. */
export const BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"];

/** Class colours for label bands: muted, non-status hues so classes never read as attack/flag/good.
 *  Binary: light grey (non-stress) vs. dark slate (stress). */
export const CLASS_BAND = ["#eef0f3", "#dfe3ea", "#e3e8f4", "#efe9f6", "#e6f0ec", "#f4eee4", "#f3e8ee", "#e9edf2"];
export const CLASS_INK = ["#a8b1bf", "#3d4a5f", "#7b8fbf", "#a58cc9", "#79a893", "#c4a07a", "#c48aa6", "#8aa4bd"];
