/* Typed client for the TrustFuse API (all routes under /api). */

export type CI = { mean: number | null; lo: number | null; hi: number | null; sd?: number | null; n?: number };

export interface TaskInfo {
  key: string;
  title: string;
  dataset: string;
  dataset_name: string;
  modalities: string[];
  classes: string[];
  window_s: number;
  stride_s: number;
  model_hz: Record<string, number>;
  headline: boolean;
  subjects_ready: number;
  subjects_total: number;
  fold_models: number;
}

export interface DatasetInfo {
  key: string;
  name: string;
  description: string;
  citation: string;
  license: string;
  url: string;
  subjects_total: number;
  subjects_ready: number;
  hours: number;
  size_mb: number;
}

export interface RunMeta {
  run_id: string;
  name: string;
  description: string;
  task: string;
  config_hash: string;
  created: string;
  seconds: number;
  status: string;
  git_commit: string | null;
  versions: Record<string, string>;
  subjects: string[];
  variants: string[];
  seeds: number[];
  n_rows: number;
  canonical: boolean;
  files: string[];
}

export interface Overview {
  tasks: TaskInfo[];
  datasets: DatasetInfo[];
  modalities: Record<string, { label: string; unit: string; native_hz: number; channels: number }>;
  results: RunMeta[];
}

export interface SubjectInfo {
  subject: string;
  duration_s: number;
  windows: number;
  class_windows: number[];
  has_model: boolean;
}

export interface SignalModality {
  t: number[];
  values: number[][];
  channels: string[];
  unit: string;
  label: string;
  hz: number;
}

export interface Segment {
  start: number;
  end: number;
  raw: number;
  name: string;
  class: number | null;
}

export interface Signals {
  subject: string;
  t0: number;
  t1: number;
  duration_s: number;
  modalities: Record<string, SignalModality>;
  segments: Segment[];
  classes: string[];
}

export interface WindowsInfo {
  subject: string;
  window_s: number;
  stride_s: number;
  start_s: number[];
  y: number[];
  segment: number[];
  classes: string[];
}

export type Fill = "zero" | "missing" | "hold";
export type AttackType = "none" | "mask" | "attenuate" | "noise";

export interface Attack {
  type: AttackType;
  modality: string | null;
  severity: number;
  fill: Fill;
  snr_db: number;
  placement: "random" | "start" | "center" | "end";
  seed: number;
}

export interface Head {
  probs: number[];
  pred: number;
  conf: number;
}

export interface ModalityEvidence {
  Q: number;
  D: number;
  U: number;
  zD: number;
  zU: number;
  anomaly: number;
  R: number;
  w: number;
  tau: number;
  median: number;
  span: number;
  unimodal: number[];
  quality: { names: string[]; values: number[]; z: number[]; reference_median: number[]; reference_scale: number[] };
}

export interface InferResult {
  window: { index: number; start_s: number; end_s: number; label: number; label_name: string };
  classes: string[];
  attack: Attack;
  attack_label: string;
  signals: Record<string, { hz: number; unit: string; label: string; channels: string[]; original: number[][]; attacked: (number | null)[][] }>;
  fixed: Head;
  defended: Head;
  clean: { fixed: Head; defended: Head };
  changed: { fixed: boolean; defended: boolean };
  correct: { fixed: boolean; defended: boolean };
  modalities: Record<string, ModalityEvidence>;
  flag: boolean;
  min_r: number;
  flag_threshold: number;
  alpha: number;
  components: string[];
  calibration: "validation" | "pooled";
  strategy: string;
  model: { split: { train: string[]; val: string[]; test: string[] }; val_f1: number; epochs: number; created: string; config_hash: string };
  trace: { step: number; label: string; ms: number }[];
}

export interface TargetedResult {
  label: number;
  clean: { fixed: Head; defended: Head };
  candidates: ({ modality: string; tv: number; p_true_fixed: number; p_true_defended: number; flag: boolean; R: number; w: number } & {
    fixed: Head;
    defended: Head;
  })[];
  targeted: string;
  oracle: string;
  fill: Fill;
}

export interface AdaptiveCandidate {
  modality: string;
  type: "mask" | "attenuate";
  fill: Fill | null;
  severity: number;
  defended_pred: number;
  fixed_pred: number;
  flag: boolean;
  suspicion: number;
  R: number;
  p_clean_def: number;
}

export interface AdaptiveResult {
  label: number;
  clean: { fixed: Head; defended: Head };
  candidates: AdaptiveCandidate[];
  chosen: AdaptiveCandidate | null;
  evasive_success: boolean;
  n_candidates: number;
  fixed_flippable: number;
  flagged: number;
}

export interface Timeline {
  subject: string;
  classes: string[];
  window_s: number;
  stride_s: number;
  start_s: number[];
  y: number[];
  segment: number[];
  fixed_pred: number[];
  defended_pred: number[];
  fixed_conf: number[];
  defended_conf: number[];
  clean_fixed_pred: number[];
  clean_defended_pred: number[];
  reliability: Record<string, number[]>;
  weights: Record<string, number[]>;
  flag: boolean[];
  affected: boolean[];
  burst: { start_s: number; end_s: number; modality: string; length: number; fill: Fill } | null;
  summary: Record<string, number>;
}

export interface Calibration {
  subject: string;
  components: string[];
  calibration: "validation" | "pooled";
  alpha: number;
  flag_threshold: number;
  modalities: Record<
    string,
    {
      tau: number;
      median: number;
      span: number;
      n_val: number;
      n_ref: number;
      anomaly_quantiles: number[];
      quality: { names: string[]; median: number[]; scale: number[]; bounds: number[] | null };
      ref: Record<string, number>;
      hist: Record<string, number[]>;
      hist_edges: Record<string, number[]>;
    }
  >;
  model: { split: { train: string[]; val: string[]; test: string[] } };
}

export interface ConditionAgg {
  variant: string;
  cond: string;
  family: string;
  attack: string | null;
  modality: string | null;
  fill: Fill | null;
  severity: number | null;
  snr_db: number | null;
  burst: number | null;
  scope: string | null;
  components: string | null;
  strategy: string | null;
  calibration?: string | null;
  n_subjects: number;
  seed_sd?: { fixed_f1: number | null; def_f1: number | null };
  [metric: string]: unknown;
}

export interface Hypothesis {
  title: string;
  statement: string;
  test: string;
  supported: boolean;
  [k: string]: unknown;
}

export interface Summary {
  run_id: string;
  task: string;
  title: string;
  classes: string[];
  modalities: string[];
  n_subjects: number;
  n_windows: number;
  baselines: { variant: string; macro_f1: CI; accuracy: CI; balanced_accuracy: CI; flag_rate?: CI }[];
  hypotheses: {
    alpha: number;
    primary_condition: string;
    H1: Hypothesis & { per_modality: any[]; supported_modalities: string[] };
    H2: Hypothesis & { friedman: any; pairwise: any[]; ranking: { modality: string; mean_drop: number; svi: number }[] };
    H3: Hypothesis & { clean_cost: CI; noninferior: boolean; margin: number; per_modality: any[]; mitigated_modalities: string[]; mean_gain_all_attacks: CI };
    H4: Hypothesis & { clean_frr: CI; frr_max: number; frr_ok: boolean; per_modality: any[]; detectable_modalities: string[] };
    secondary: {
      targeted_vs_random?: { random_f1: CI; targeted_f1: CI; difference: CI; p: number; rank_biserial: number | null; statement: string };
      attackers: { attacker: string; fixed_f1: CI; def_f1: CI; flag_rate: CI; evasive_def_flip: CI; fixed_asr: CI; def_asr: CI }[];
    };
  };
  moddrop_comparison: { family: string; fusion_fixed: CI; fusion_defended: CI; moddrop_fixed: CI; moddrop_defended: CI }[] | null;
  forest: { modality: string; subject: string; fixed_f1: number; def_f1: number; gain: number }[];
  svi: Record<string, number>;
  defense?: { components: string[]; calibration: string; strategy: string; alpha: number };
  operating_points?: OperatingPoint[];
}

/** The primary defense or one ablation of it, on clean windows and under full suppression of each modality. */
export interface OperatingPoint {
  components: string;
  strategy: string;
  calibration: string;
  primary: boolean;
  clean_frr: CI;
  clean_fixed_f1: CI;
  clean_def_f1: CI;
  full_fixed_f1: CI;
  full_def_f1: CI;
  full_gain: CI;
  full_flag_rate: CI;
  full_auroc: CI;
  per_modality: Record<string, { fixed_f1: CI; def_f1: CI }>;
}

export interface Detection {
  curves: Record<string, { auroc: number; fpr: number[]; tpr: number[] }>;
  matrix: { family: string; modality: string | null; auroc?: number; auroc_Q?: number; auroc_D?: number; auroc_U?: number }[];
  clean_frr: CI;
  alpha: number;
}

export interface JobPublic {
  id: string;
  name: string;
  kind: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  stage: string;
  progress: number;
  created: number;
  started: number | null;
  finished: number | null;
  run_id: string | null;
  error: string | null;
  task: string;
  n_events: number;
}

export interface JobEvent {
  seq: number;
  t: number;
  type: "log" | "progress" | "epoch" | "stage" | "status";
  data: Record<string, any>;
}

export interface Preset {
  key: string;
  label: string;
  summary: string;
  minutes: number;
  config: Record<string, any>;
  config_hash: string;
  train_jobs: number;
  test_subjects: string[] | null;
}

export interface SystemInfo {
  version: string;
  python: string;
  torch: string;
  threads: number;
  cpus: number;
  memory: { rss_bytes: number | null; cgroup_limit_bytes: number | null; cgroup_used_bytes: number | null };
  uptime_s: number;
  in_docker: boolean;
  jobs_active: JobPublic[];
  cached_models: number;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, msg);
  }
  return res.json() as Promise<T>;
}

const post = <T>(path: string, body: unknown) => request<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  health: () => request<{ status: string; version: string }>("/health"),
  system: () => request<SystemInfo>("/system"),
  overview: () => request<Overview>("/overview"),
  subjects: (task: string) => request<SubjectInfo[]>(`/tasks/${task}/subjects`),
  signals: (task: string, subject: string, t0?: number, t1?: number, points = 1500) =>
    request<Signals>(
      `/tasks/${task}/subjects/${subject}/signals?points=${points}${t0 != null ? `&t0=${t0}` : ""}${t1 != null ? `&t1=${t1}` : ""}`,
    ),
  windows: (task: string, subject: string) => request<WindowsInfo>(`/tasks/${task}/subjects/${subject}/windows`),
  model: (task: string, variant: string, seed: number, subject: string) =>
    request<{ history: { epoch: number; loss: number; val_f1: number; val_acc: number }[]; val_f1: number; split: any; parameters: number; bytes: number; created: string; train_cfg: any }>(
      `/models/${task}/${variant}/${seed}/${subject}`,
    ),
  infer: (body: { task: string; subject: string; window: number; variant?: string; attack: Attack; components?: string[]; calibration?: string; strategy?: string }) =>
    post<InferResult>("/infer", body),
  targeted: (body: { task: string; subject: string; window: number; fill: Fill; variant?: string }) => post<TargetedResult>("/attack/targeted", body),
  adaptive: (body: { task: string; subject: string; window: number; variant?: string }) => post<AdaptiveResult>("/attack/adaptive", body),
  timeline: (body: { task: string; subject: string; variant?: string; burst?: { modality: string; start_index: number; length: number; fill: Fill } | null }) =>
    post<Timeline>("/timeline", body),
  calibration: (task: string, subject: string, components = "QDU", calibration = "validation") =>
    request<Calibration>(`/calibration/${task}/${subject}?components=${components}&calibration=${calibration}`),
  results: () => request<RunMeta[]>("/results"),
  summary: (run: string) => request<Summary>(`/results/${run}/summary`),
  conditions: (run: string) => request<{ run_id: string; task: string; conditions: ConditionAgg[] }>(`/results/${run}/conditions`),
  detection: (run: string) => request<Detection>(`/results/${run}/detection`),
  runMeta: (run: string) => request<RunMeta>(`/results/${run}/run`),
  manifestConditions: (run: string) => request<{ conditions: string[]; fields: string[]; n_windows: number }>(`/results/${run}/manifest/conditions`),
  manifestRows: (run: string, cond: string, limit = 200, offset = 0) =>
    request<{ condition: string; total: number; offset: number; rows: Record<string, number | string>[] }>(
      `/results/${run}/manifest/rows?cond=${encodeURIComponent(cond)}&limit=${limit}&offset=${offset}`,
    ),
  presets: () => request<Preset[]>("/experiments/presets"),
  validate: (body: { preset?: string; overrides?: Record<string, unknown> }) =>
    post<{ ok: boolean; config_hash: string; train_jobs: number; yaml: string }>("/experiments/validate", body),
  submit: (body: { preset?: string; overrides?: Record<string, unknown> }) => post<JobPublic>("/jobs", body),
  jobs: () => request<JobPublic[]>("/jobs"),
  job: (id: string) => request<JobPublic & { config: any; events: JobEvent[] }>(`/jobs/${id}`),
  cancel: (id: string) => request<JobPublic>(`/jobs/${id}`, { method: "DELETE" }),
};

export const downloadUrl = {
  file: (run: string, name: string) => `/api/results/${run}/files/${name}`,
  bundle: (run: string) => `/api/results/${run}/bundle.zip`,
  manifestCsv: (run: string, cond: string) => `/api/results/${run}/manifest/csv?cond=${encodeURIComponent(cond)}`,
  latex: (run: string) => `/api/reports/${run}/latex`,
};

export async function downloadPost(url: string, filename: string) {
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) throw new ApiError(res.status, res.statusText);
  const blob = await res.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}
