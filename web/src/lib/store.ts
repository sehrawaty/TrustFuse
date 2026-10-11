import { create } from "zustand";
import type { Attack } from "./api";

export const DEFAULT_ATTACK: Attack = {
  type: "mask",
  modality: "eda",
  severity: 1,
  fill: "zero",
  snr_db: 10,
  placement: "center",
  seed: 0,
};

interface AppState {
  task: string;
  subject: Record<string, string>;
  window: Record<string, number>;
  attack: Attack;
  variant: "fusion" | "moddrop";
  components: string[];
  run: Record<string, string>;
  demoStep: number | null;
  setTask: (t: string) => void;
  setSubject: (task: string, s: string) => void;
  setWindow: (key: string, w: number) => void;
  setAttack: (a: Partial<Attack>) => void;
  setVariant: (v: "fusion" | "moddrop") => void;
  toggleComponent: (c: string) => void;
  setComponents: (c: string[]) => void;
  setRun: (task: string, run: string) => void;
  setDemoStep: (s: number | null) => void;
}

const remembered = (() => {
  try {
    return JSON.parse(localStorage.getItem("trustfuse:prefs") ?? "{}");
  } catch {
    return {};
  }
})();

export const useApp = create<AppState>((set, get) => ({
  task: remembered.task ?? "wesad_binary",
  subject: remembered.subject ?? { wesad_binary: "S2", wesad_3class: "S2", dalia_activity: "S1" },
  window: {},
  attack: DEFAULT_ATTACK,
  variant: "fusion",
  components: ["Q"],
  run: {},
  demoStep: null,
  setTask: (task) => {
    const mods = task.startsWith("dalia") ? ["bvp", "acc"] : ["bvp", "acc", "eda", "temp"];
    const a = get().attack;
    set({ task, attack: a.modality && !mods.includes(a.modality) ? { ...a, modality: mods[0] } : a });
    persist();
  },
  setSubject: (task, s) => {
    set({ subject: { ...get().subject, [task]: s } });
    persist();
  },
  setWindow: (key, w) => set({ window: { ...get().window, [key]: w } }),
  setAttack: (a) => set({ attack: { ...get().attack, ...a } }),
  setVariant: (variant) => set({ variant }),
  toggleComponent: (c) => {
    const cur = get().components;
    set({ components: cur.includes(c) ? cur.filter((x) => x !== c) : [...cur, c].sort((a, b) => "QDU".indexOf(a) - "QDU".indexOf(b)) });
  },
  setComponents: (components) => set({ components }),
  setRun: (task, run) => set({ run: { ...get().run, [task]: run } }),
  setDemoStep: (demoStep) => set({ demoStep }),
}));

function persist() {
  try {
    const { task, subject } = useApp.getState();
    localStorage.setItem("trustfuse:prefs", JSON.stringify({ task, subject }));
  } catch {
    /* storage unavailable */
  }
}

export const windowKey = (task: string, subject: string) => `${task}:${subject}`;

/** Initial window for a subject: a few windows into the first stress segment (the interesting case). */
export function defaultWindow(w?: { y: number[] }) {
  if (!w) return 0;
  const i = w.y.findIndex((c) => c === 1);
  return i >= 0 ? Math.min(i + 8, w.y.length - 1) : 0;
}
