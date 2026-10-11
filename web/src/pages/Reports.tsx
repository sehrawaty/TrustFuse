import { useMutation, useQuery } from "@tanstack/react-query";
import { Check, Copy, Download, FileText, Image } from "lucide-react";
import { useState } from "react";
import Tex from "../components/Tex";
import { Button, Card, CardHeader, Empty, ErrorNote, PageHeader, SectionTitle, Select, Skeleton } from "../components/ui";
import { useRun } from "../components/useRun";
import { downloadPost, downloadUrl } from "../lib/api";
import { useApp } from "../lib/store";

const FILES: [string, string][] = [
  ["summary.json", "Baselines, hypotheses H1–H4, secondary analyses"],
  ["conditions.json", "Every condition with subject-bootstrap CIs"],
  ["detection.json", "Pooled ROC curves and detectability matrix"],
  ["subjects.csv.gz", "Subject-level table (unit of analysis)"],
  ["rows.csv.gz", "Raw measurements: variant × seed × subject × condition"],
  ["manifest.npz", "Per-window decisions of every window-level attack"],
  ["bursts.csv.gz", "Per-window decisions of every burst-affected window"],
  ["calibration.json", "Frozen reliability calibration of each fold"],
  ["config.yaml", "Exact experiment configuration"],
  ["run.json", "Provenance: config hash, versions, commit, timing"],
];

export default function Reports() {
  const { task } = useApp();
  const { runId, runs, setRun } = useRun();
  const latex = useQuery({
    queryKey: ["latex", runId],
    queryFn: async () => (await fetch(downloadUrl.latex(runId!))).text(),
    enabled: !!runId,
  });
  const fig = useMutation({ mutationFn: (fmt: string) => downloadPost(`/api/reports/${runId}/figures?fmt=${fmt}`, `${runId}_figures_${fmt}.zip`) });
  const [copied, setCopied] = useState(false);

  return (
    <div>
      <PageHeader
        eyebrow="Evaluate"
        title="Reports & method"
        description="Paper-ready figures and tables, generated server-side from the selected result set, using the same validated palette as this interface."
        actions={
          runs.length > 0 && (
            <Select value={runId ?? ""} onChange={setRun} options={runs.map((r) => ({ value: r.run_id, label: r.canonical ? `${r.run_id} · canonical` : r.run_id }))} className="w-[260px]" />
          )
        }
      />
      {!runId ? (
        <Card>
          <Empty title={`No result set for ${task}`} />
        </Card>
      ) : (
        <div className="grid grid-cols-12 gap-4">
          <Card className="col-span-12 xl:col-span-5">
            <CardHeader eyebrow="Figures" title="Export the paper figures" sub="Severity sweeps, vulnerability heatmap, attacker ladder, detection ROC, per-subject forest plot, burst curves. Includes tables.tex." />
            <div className="flex flex-wrap gap-2 p-5">
              {(["pdf", "svg", "png"] as const).map((f) => (
                <Button key={f} variant={f === "pdf" ? "primary" : "secondary"} icon={f === "png" ? <Image className="h-3.5 w-3.5" /> : <Download className="h-3.5 w-3.5" />} loading={fig.isPending && fig.variables === f} onClick={() => fig.mutate(f)}>
                  {f.toUpperCase()}
                </Button>
              ))}
            </div>
            {fig.isError && <div className="px-5 pb-4"><ErrorNote error={fig.error} /></div>}
            <div className="border-t border-line">
              <div className="eyebrow px-5 pt-4">Raw result files</div>
              <ul className="divide-y divide-line px-2 py-2">
                {FILES.map(([f, d]) => (
                  <li key={f} className="flex items-center justify-between px-3 py-1.5 text-[12.5px]">
                    <span>
                      <span className="font-mono text-[11.5px]">{f}</span>
                      <span className="ml-2 text-muted">{d}</span>
                    </span>
                    <a className="link shrink-0" href={downloadUrl.file(runId, f)}>
                      download
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </Card>
          <Card className="col-span-12 xl:col-span-7">
            <CardHeader
              eyebrow="LaTeX"
              title="Tables for the paper"
              actions={
                <Button
                  size="sm"
                  icon={copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                  onClick={() => {
                    navigator.clipboard?.writeText(latex.data ?? "");
                    setCopied(true);
                    window.setTimeout(() => setCopied(false), 1500);
                  }}
                >
                  {copied ? "Copied" : "Copy"}
                </Button>
              }
            />
            {latex.data ? (
              <pre className="max-h-[440px] overflow-auto whitespace-pre-wrap p-5 font-mono text-[11px] leading-relaxed text-ink-2">{latex.data}</pre>
            ) : (
              <Skeleton className="m-5 h-72" />
            )}
          </Card>
        </div>
      )}

      <SectionTitle>Method</SectionTitle>
      <div className="grid grid-cols-12 gap-4">
        <Card className="col-span-12 xl:col-span-8">
          <div className="space-y-5 p-6 text-[13.5px] leading-relaxed text-ink-2">
            <Section title="Threat model">
              An attacker with access to the sensing path suppresses exactly one modality m of a synchronized window at inference time. The attack
              transform <Tex>{String.raw`A_m`}</Tex> changes only <Tex>{String.raw`x_m`}</Tex>; every other stream <Tex>{String.raw`x_j, j\neq m`}</Tex> is passed through unchanged
              (enforced by unit tests). The primitives are full or partial masking (zero, missing or frozen value), amplitude attenuation, stream-level bursts over B consecutive
              windows, and additive noise. The attackers are random, targeted (label-free and query-based), oracle (label-aware) and adaptive
              (aware of the defense).
            </Section>
            <Section title="Protocol and leakage control">
              Leave-one-subject-out over 15 subjects per dataset. In each fold, two other subjects are held out for validation (early stopping and calibration), and normalisation uses
              training subjects only. The fixed and reliability-aware pipelines share one frozen checkpoint and receive identical attacked windows. The
              severity grid (10–80 % and full), burst lengths (1, 2, 5, 10), the flag level α and the non-inferiority margin are fixed in the configuration before test
              evaluation.
            </Section>
            <Section title="Reliability-aware fusion">
              Evidence per modality comprises signal quality <Tex>{"Q_m"}</Tex>, disagreement with the other unimodal heads <Tex>{"D_m"}</Tex>, and MC-dropout uncertainty{" "}
              <Tex>{"U_m"}</Tex>. Their calibrated sum is ranked against clean validation windows to produce a conformal p-value <Tex>{"R_m"}</Tex>. The flag fires when{" "}
              <Tex>{String.raw`\min_m R_m\le\alpha/M`}</Tex>. The gate <Tex>{"w_m"}</Tex> blends <Tex>{"h_m"}</Tex> towards the clean reference embedding{" "}
              <Tex>{String.raw`\mu_m`}</Tex> before the frozen classifier. Every scaling parameter and threshold is fitted on clean validation subjects only and frozen before attack
              evaluation. A variant that references signal quality to all non-test subjects is evaluated separately as an extension (Results, Ablations).
            </Section>
            <Section title="Statistics">
              The subject is the unit of analysis. Seeds are averaged within each subject. 95 % confidence intervals use a subject-cluster bootstrap with 2,000 resamples.
              H1 and H3 use one-sided paired Wilcoxon signed-rank tests with Holm correction. H2 uses a Friedman test with Kendall's W. H4 uses a one-sample Wilcoxon of the
              per-subject AUROC against 0.5, plus the clean false-rejection rate. Effect sizes are matched-pairs rank-biserial r and Cohen's d<sub>z</sub>.
            </Section>
            <Section title="Positioning">
              The contribution is a security-oriented empirical characterisation. Missing-modality handling, reliability weighting and fusion are not individually
              novel. What is new is treating deliberate suppression as an attack primitive, and measuring how much damage it does and how detectable it is. Every number in this
              application comes from an executed run; a hypothesis that the data do not support is reported as such.
            </Section>
          </div>
        </Card>
        <Card className="col-span-12 xl:col-span-4">
          <CardHeader eyebrow="Reproduce" title="From a fresh clone" />
          <div className="space-y-3 p-5 text-[12.5px]">
            {[
              [".\\config-everything", "Build images, verify data checksums, run the smoke test"],
              [".\\run", "Fresh start of the stack and open the browser"],
              [".\\stop", "Stop everything (-All also quits Docker Desktop)"],
              ["python -m trustfuse train wesad_binary --workers 3", "Retrain every fold model (outside Docker)"],
              ["python -m trustfuse evaluate wesad_binary --workers 3", "Recompute the canonical result set"],
            ].map(([c, d]) => (
              <div key={c}>
                <code className="block rounded-md bg-sunken px-2.5 py-1.5 font-mono text-[11.5px] text-ink ring-1 ring-line">{c}</code>
                <div className="mt-1 text-muted">{d}</div>
              </div>
            ))}
            <div className="flex items-start gap-2 border-t border-line pt-3 text-muted">
              <FileText className="mt-0.5 h-4 w-4 shrink-0" />
              Datasets: WESAD (Schmidt et al., ICMI 2018) and PPG-DaLiA (Reiss et al., Sensors 2019). See data/DATA_LICENSE.md for terms.
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="mb-1 text-[14px] font-semibold text-ink">{title}</h3>
      <p>{children}</p>
    </div>
  );
}
