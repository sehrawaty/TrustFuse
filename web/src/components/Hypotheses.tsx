import clsx from "clsx";
import { CheckCircle2, CircleDashed, XCircle } from "lucide-react";
import type { Summary } from "../lib/api";
import { num, pval, signed } from "../lib/format";
import { MODALITY_LABEL } from "../lib/palette";
import { Badge, CiText, Tip } from "./ui";

function Verdict({ ok }: { ok: boolean }) {
  return ok ? (
    <Badge tone="good" icon={<CheckCircle2 className="h-3 w-3" />}>
      Supported
    </Badge>
  ) : (
    <Badge tone="neutral" icon={<XCircle className="h-3 w-3" />}>
      Not supported
    </Badge>
  );
}

export function HypothesisCards({ s, compact }: { s?: Summary; compact?: boolean }) {
  if (!s)
    return (
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {["H1", "H2", "H3", "H4"].map((h) => (
          <div key={h} className="card flex h-40 items-center justify-center gap-2 text-[12.5px] text-muted">
            <CircleDashed className="h-4 w-4" /> {h} awaiting results
          </div>
        ))}
      </div>
    );
  const H = s.hypotheses;
  const items = [
    {
      k: "H1",
      h: H.H1,
      key: (
        <>
          largest drop{" "}
          {(() => {
            const best = [...H.H1.per_modality].sort((a, b) => (b.drop.mean ?? 0) - (a.drop.mean ?? 0))[0];
            return best ? (
              <>
                <b>{MODALITY_LABEL[best.modality]}</b> <span className="num">{signed(-(best.drop.mean ?? 0))}</span> F1
              </>
            ) : null;
          })()}
        </>
      ),
      detail: `significant for ${H.H1.supported_modalities.map((m) => MODALITY_LABEL[m]).join(", ") || "no modality"}`,
    },
    {
      k: "H2",
      h: H.H2,
      key: (
        <>
          ranking{" "}
          <b>{H.H2.ranking.map((r) => MODALITY_LABEL[r.modality]).join(" > ")}</b>
        </>
      ),
      detail: H.H2.friedman?.kendall_w != null ? `Friedman p ${pval(H.H2.friedman.p)}, Kendall's W ${num(H.H2.friedman.kendall_w, 2)}` : `p ${pval(H.H2.friedman?.p)}`,
    },
    {
      k: "H3",
      h: H.H3,
      key: (
        <>
          mean gain under attack <CiText ci={H.H3.mean_gain_all_attacks} />
        </>
      ),
      detail: `clean cost ${signed(-(H.H3.clean_cost.mean ?? 0))} F1 (margin ${H.H3.margin}) · mitigated: ${H.H3.mitigated_modalities.map((m) => MODALITY_LABEL[m]).join(", ") || "none"}`,
    },
    {
      k: "H4",
      h: H.H4,
      key: (
        <>
          clean false-rejection <CiText ci={H.H4.clean_frr} pct />
        </>
      ),
      detail: `AUROC > 0.5 for ${H.H4.detectable_modalities.map((m) => MODALITY_LABEL[m]).join(", ") || "none"} (limit ${(H.H4.frr_max * 100).toFixed(0)}% FRR)`,
    },
  ];
  return (
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
      {items.map((it) => (
        <div key={it.k} className={clsx("card flex flex-col p-4", it.h.supported && "ring-1 ring-good/20")}>
          <div className="mb-2 flex items-center justify-between">
            <span className="num text-[12px] font-semibold text-muted">{it.k}</span>
            <Verdict ok={it.h.supported} />
          </div>
          <div className="text-[14px] font-semibold text-ink">{it.h.title}</div>
          {!compact && <p className="mt-1 text-[12.5px] leading-relaxed text-ink-2">{it.h.statement}</p>}
          <div className="mt-auto pt-3 text-[12.5px] text-ink-2">{it.key}</div>
          <Tip content={it.h.test}>
            <div className="mt-1 cursor-help text-[11.5px] text-muted">{it.detail}</div>
          </Tip>
        </div>
      ))}
    </div>
  );
}
