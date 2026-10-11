import { useQuery } from "@tanstack/react-query";
import { ArrowRight, BookOpen, Database, Lock, Radar } from "lucide-react";
import { Link } from "react-router-dom";
import { HypothesisCards } from "../components/Hypotheses";
import PipelineDiagram from "../components/PipelineDiagram";
import { Card, CardHeader, PageHeader, SectionTitle, Skeleton, Stat } from "../components/ui";
import { useRun, useSummary } from "../components/useRun";
import { api } from "../lib/api";
import { useApp } from "../lib/store";

const CONTRIBUTIONS = [
  {
    title: "Suppression as a security primitive",
    body: "Selective Modality Suppression (SMSA) is formalised as an inference-time attack with masking, attenuation and burst variants, and attackers of increasing capability: random, targeted, oracle and defense-aware.",
  },
  {
    title: "Subject-diverse conformal reliability",
    body: "Per-modality evidence is turned into calibrated p-values, which give a low-confidence flag and a fusion gate on the same frozen model. Calibrating against many people, not just two, is what keeps false alarms on new users near the target rate.",
  },
  {
    title: "Matched security evaluation",
    body: "Fixed and reliability-aware fusion receive identical attacked windows. Results are subject-independent (LOSO), use bootstrap CIs, and test pre-registered hypotheses with traceable per-window manifests.",
  },
  {
    title: "Honest threat coverage",
    body: "A defense-aware adaptive attacker and a modality-dropout baseline test whether robustness to generic missingness also gives security against strategic suppression.",
  },
];

export default function Overview() {
  const { task } = useApp();
  const ov = useQuery({ queryKey: ["overview"], queryFn: api.overview });
  const taskInfo = ov.data?.tasks.find((t) => t.key === task);
  const { runId } = useRun();
  const summary = useSummary(runId);
  const s = summary.data;
  const fusion = s?.baselines.find((b) => b.variant === "fusion");
  const defended = s?.baselines.find((b) => b.variant === "fusion+defense");
  const targeted = s?.hypotheses.secondary.attackers.find((a) => a.attacker === "targeted");
  const auroc = s?.hypotheses.H4.per_modality;
  const meanAuroc = auroc?.length ? auroc.reduce((a, r) => a + (r.auroc.mean ?? 0), 0) / auroc.length : null;

  return (
    <div>
      <PageHeader
        eyebrow="Final-year project · VIT SCOPE"
        title="Security evaluation of intentional modality suppression in wearable health AI"
        description="Robustness to missing data is not security. TrustFuse treats deliberate suppression of one sensor stream as an attack, measures how much each modality matters, and tests a reliability-aware fusion defense on the same frozen model."
        actions={
          <>
            <Link to="/attack" className="inline-flex h-8 items-center gap-1.5 rounded-lg bg-ink px-3 text-[13px] font-medium text-white hover:bg-ink-2">
              <Radar className="h-3.5 w-3.5" /> Open Attack Lab
            </Link>
            <Link to="/reports" className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-line-strong bg-surface px-3 text-[13px] font-medium hover:bg-sunken">
              <BookOpen className="h-3.5 w-3.5" /> Method
            </Link>
          </>
        }
      />

      <div className="mb-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Card className="card-pad">
          <Stat
            label="Clean macro-F1 · fixed fusion"
            tone="fixed"
            value={fusion?.macro_f1.mean != null ? fusion.macro_f1.mean.toFixed(3) : "—"}
            sub={fusion ? `95% CI ${fusion.macro_f1.lo?.toFixed(3)}–${fusion.macro_f1.hi?.toFixed(3)} · subject bootstrap` : "awaiting canonical results"}
          />
        </Card>
        <Card className="card-pad">
          <Stat label="Under targeted suppression" tone="attack" value={targeted ? (targeted.fixed_f1.mean ?? 0).toFixed(3) : "—"} sub="fixed fusion macro-F1, one modality per window" />
        </Card>
        <Card className="card-pad">
          <Stat label="Same attack · reliability-aware" tone="defended" value={targeted ? (targeted.def_f1.mean ?? 0).toFixed(3) : "—"} sub={defended ? `clean macro-F1 ${defended.macro_f1.mean?.toFixed(3)}` : "defended macro-F1"} />
        </Card>
        <Card className="card-pad">
          <Stat label="Suppression detection" tone="good" value={meanAuroc != null ? meanAuroc.toFixed(3) : "—"} sub={s ? `mean AUROC, full masking · ${s.n_subjects} held-out subjects` : "AUROC"} />
        </Card>
      </div>

      <Card className="mb-4">
        <CardHeader eyebrow="System architecture" title="Where the attack enters and where the defense acts" sub="Interactive rebuild of report Fig. 2. Click a stage to open the matching page." />
        <div className="p-5">
          <PipelineDiagram modalities={taskInfo?.modalities ?? ["bvp", "acc", "eda", "temp"]} />
        </div>
      </Card>

      <SectionTitle aside={runId && <Link to="/results" className="link flex items-center gap-1 text-[12.5px]">Full evaluation <ArrowRight className="h-3.5 w-3.5" /></Link>}>
        Pre-registered hypotheses {s && <span className="ml-1 text-[12.5px] font-normal text-muted">· {s.title} · {s.n_subjects} LOSO folds</span>}
      </SectionTitle>
      {summary.isLoading ? <Skeleton className="h-40" /> : <HypothesisCards s={s} />}

      <SectionTitle>What this project contributes</SectionTitle>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {CONTRIBUTIONS.map((c, i) => (
          <div key={c.title} className="card p-4">
            <div className="num mb-2 text-[11px] text-faint">0{i + 1}</div>
            <div className="text-[13.5px] font-semibold">{c.title}</div>
            <p className="mt-1.5 text-[12.5px] leading-relaxed text-ink-2">{c.body}</p>
          </div>
        ))}
      </div>

      <SectionTitle>Data</SectionTitle>
      <div className="grid gap-3 md:grid-cols-2">
        {(ov.data?.datasets ?? []).map((d) => (
          <div key={d.key} className="card p-4">
            <div className="mb-2 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Database className="h-4 w-4 text-muted" />
                <span className="text-[14px] font-semibold">{d.name}</span>
              </div>
              <span className="chip">
                <Lock className="h-3 w-3" /> {d.license}
              </span>
            </div>
            <p className="text-[12.5px] leading-relaxed text-ink-2">{d.description}</p>
            <div className="mt-3 flex gap-6 text-[12px]">
              <span>
                <span className="num text-ink">
                  {d.subjects_ready}/{d.subjects_total}
                </span>{" "}
                <span className="text-muted">subjects</span>
              </span>
              <span>
                <span className="num text-ink">{d.hours}</span> <span className="text-muted">hours</span>
              </span>
              <span>
                <span className="num text-ink">{d.size_mb} MB</span> <span className="text-muted">processed</span>
              </span>
            </div>
            <p className="mt-3 border-t border-line pt-2 text-[11.5px] leading-relaxed text-muted">{d.citation}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
