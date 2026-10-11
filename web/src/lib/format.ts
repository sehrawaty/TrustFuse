export const pct = (v: number | null | undefined, digits = 1) =>
  v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const num = (v: number | null | undefined, digits = 3) =>
  v == null || Number.isNaN(v) ? "—" : v.toFixed(digits);

export const signed = (v: number | null | undefined, digits = 3) => {
  if (v == null || Number.isNaN(v)) return "—";
  if (Math.abs(v) < 0.5 * 10 ** -digits) return (0).toFixed(digits); // no "−0.000"
  return `${v > 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;
};

export const pval = (p: number | null | undefined) =>
  p == null || Number.isNaN(p) ? "n/a" : p < 0.001 ? "< 0.001" : p.toFixed(3);

export const seconds = (s: number) => {
  if (s < 60) return `${s.toFixed(0)} s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s % 60);
  if (m < 60) return `${m}:${String(r).padStart(2, "0")} min`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
};

export const clock = (s: number) => {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${m}:${String(sec).padStart(2, "0")}`;
};

export const bytes = (b: number | null | undefined) => {
  if (b == null) return "—";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  let v = b;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${u[i]}`;
};

export const ago = (t: number) => {
  const d = Date.now() / 1000 - t;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  return new Date(t * 1000).toLocaleDateString();
};

export const titleCase = (s: string) => s.replace(/(^|[\s_-])(\w)/g, (_, a, b) => `${a === "_" ? " " : a}${b.toUpperCase()}`);
