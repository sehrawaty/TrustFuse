import clsx from "clsx";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";

const ACTORS = ["Sensor stream", "Attack layer", "Model", "Reliability engine", "Fusion / decision"];
// (from actor, to actor) for each of the report's sequence-diagram steps (Fig. 6)
const ARROWS: [number, number][] = [
  [0, 2], [0, 2], [1, 2], [1, 3], [2, 4], [2, 4], [2, 4], [3, 0],
];

/** Animated version of the runtime sequence (report Fig. 6) driven by the live API trace. */
export default function SequenceTrace({ trace, version, attacked }: { trace: { step: number; label: string; ms: number }[]; version: number; attacked: boolean }) {
  const [active, setActive] = useState(trace.length);
  useEffect(() => {
    setActive(0);
    let i = 0;
    const id = window.setInterval(() => {
      i += 1;
      setActive(i);
      if (i >= trace.length) window.clearInterval(id);
    }, 110);
    return () => window.clearInterval(id);
  }, [version, trace.length]);
  const total = trace.reduce((a, s) => a + s.ms, 0);
  const colX = (i: number) => 8 + i * 23;
  return (
    <div>
      <div className="mb-2 grid grid-cols-5 text-center text-[11px] font-medium text-muted">
        {ACTORS.map((a) => (
          <div key={a}>{a}</div>
        ))}
      </div>
      <div className="relative">
        <div className="pointer-events-none absolute inset-0 grid grid-cols-5">
          {ACTORS.map((a) => (
            <div key={a} className="mx-auto w-px bg-line" />
          ))}
        </div>
        <div className="relative space-y-1">
          {trace.map((s, i) => {
            const [a, b] = ARROWS[i] ?? [0, 4];
            const left = Math.min(colX(a), colX(b));
            const width = Math.abs(colX(b) - colX(a));
            const on = i < active;
            const muted = i === 2 && !attacked;
            return (
              <div key={s.step} className="relative h-6">
                <motion.div
                  className={clsx("absolute top-1/2 h-[1.5px]", on ? (i === 2 && attacked ? "bg-attack" : i === 7 ? "bg-faint" : "bg-ink") : "bg-transparent")}
                  style={{ left: `${left}%`, width: `${width}%` }}
                  initial={false}
                  animate={{ opacity: on ? (muted ? 0.3 : 1) : 0 }}
                />
                <div
                  className={clsx(
                    "absolute top-0 -translate-x-1/2 whitespace-nowrap rounded bg-surface px-1.5 text-[11px] transition-opacity",
                    on ? "opacity-100" : "opacity-0",
                    muted ? "text-faint" : "text-ink-2",
                  )}
                  style={{ left: `${left + width / 2}%` }}
                >
                  <span className="num mr-1 text-faint">{s.step}.</span>
                  {s.label}
                  <span className="num ml-1.5 text-faint">{s.ms.toFixed(1)} ms</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <div className="mt-2 text-right text-[11px] text-muted">
        end-to-end <span className="num text-ink">{total.toFixed(1)} ms</span> on CPU
      </div>
    </div>
  );
}
