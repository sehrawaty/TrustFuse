import clsx from "clsx";
import { useMemo } from "react";
import { CLASS_BAND, CLASS_INK } from "../lib/palette";
import { clock } from "../lib/format";

/** Compact clickable strip of every evaluation window of a subject, coloured by true class. */
export default function WindowStrip({
  start,
  y,
  classes,
  value,
  onChange,
  marks,
  windowS,
}: {
  start: number[];
  y: number[];
  classes: string[];
  value: number;
  onChange: (i: number) => void;
  marks?: (boolean | null)[];
  windowS: number;
}) {
  const t0 = start[0] ?? 0;
  const t1 = (start[start.length - 1] ?? 0) + windowS;
  const span = Math.max(t1 - t0, 1);
  const cells = useMemo(() => start.map((s, i) => ({ i, left: ((s - t0) / span) * 100, cls: y[i] })), [start, y, t0, span]);
  const sel = cells[value];
  return (
    <div>
      <div
        className="relative h-9 cursor-pointer select-none overflow-hidden rounded-md border border-line bg-sunken"
        onClick={(e) => {
          const r = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
          const t = t0 + ((e.clientX - r.left) / r.width) * span;
          let best = 0;
          let bd = Infinity;
          start.forEach((s, i) => {
            const d = Math.abs(s + windowS / 2 - t);
            if (d < bd) {
              bd = d;
              best = i;
            }
          });
          onChange(best);
        }}
        role="slider"
        aria-valuemin={0}
        aria-valuemax={start.length - 1}
        aria-valuenow={value}
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight") onChange(Math.min(start.length - 1, value + 1));
          if (e.key === "ArrowLeft") onChange(Math.max(0, value - 1));
        }}
      >
        {cells.map((c) => (
          <div
            key={c.i}
            className="absolute inset-y-0"
            style={{ left: `${c.left}%`, width: `${Math.max((windowS / span) * 100 * 0.3, 0.25)}%`, background: CLASS_INK[c.cls], opacity: 0.55 }}
          />
        ))}
        {marks?.map((m, i) =>
          m ? <div key={`m${i}`} className="absolute bottom-0 h-1.5 w-[2px] bg-attack" style={{ left: `${cells[i].left}%` }} /> : null,
        )}
        {sel && (
          <div
            className="absolute inset-y-0 rounded-sm border-2 border-ink bg-ink/5"
            style={{ left: `${sel.left}%`, width: `${Math.max((windowS / span) * 100, 0.8)}%` }}
          />
        )}
      </div>
      <div className="mt-1.5 flex items-center justify-between text-[11px] text-muted">
        <span className="num">{clock(t0)}</span>
        <div className="flex gap-3">
          {classes.map((c, i) => (
            <span key={c} className="flex items-center gap-1">
              <span className={clsx("h-2 w-2 rounded-sm")} style={{ background: CLASS_INK[i], opacity: 0.6 }} />
              {c}
            </span>
          ))}
        </div>
        <span className="num">{clock(t1)}</span>
      </div>
    </div>
  );
}

export { CLASS_BAND };
