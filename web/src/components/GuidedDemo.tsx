import { AnimatePresence, motion } from "framer-motion";
import { ArrowLeft, ArrowRight, X } from "lucide-react";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { DEMO } from "../lib/demo";
import { useApp, windowKey } from "../lib/store";
import { Button, Kbd } from "./ui";

interface Step {
  title: string;
  body: string;
  route: string;
  apply?: () => void;
}

/** Presentation mode: a scripted walk-through that drives the real application state. */
export default function GuidedDemo() {
  const { demoStep, setDemoStep, setTask, setSubject, setWindow, setAttack, setComponents, setCalibration, setVariant } = useApp();
  const navigate = useNavigate();

  const base = () => {
    setTask(DEMO.task);
    setSubject(DEMO.task, DEMO.subject);
    setWindow(windowKey(DEMO.task, DEMO.subject), DEMO.window);
    setComponents(["Q", "D", "U"]);
    setCalibration("validation");
    setVariant("fusion");
  };

  const steps: Step[] = [
    {
      title: "The question",
      body: "Wearable health AI fuses BVP, motion, EDA and skin temperature. Models are usually tested against accidental sensor loss. TrustFuse asks a security question instead: what happens when an attacker deliberately suppresses the one stream that matters most?",
      route: "/",
    },
    {
      title: "A clean, held-out window",
      body: `Subject ${DEMO.subject} was never seen in training (leave-one-subject-out). With all four streams intact, the standard fusion model and the reliability-aware pipeline agree.`,
      route: "/attack",
      apply: () => {
        base();
        setAttack({ type: "none" });
      },
    },
    {
      title: `Suppress ${DEMO.modalityLabel}`,
      body: `The attacker zeroes only the ${DEMO.modalityLabel} stream and leaves the other three untouched. The standard fusion model still produces a confident answer, but the decision has changed.`,
      route: "/attack",
      apply: () => {
        base();
        setAttack({ type: "mask", modality: DEMO.modality, severity: 1, fill: "zero", placement: "center" });
      },
    },
    {
      title: "The reliability engine notices",
      body: "Signal quality (Q), cross-modal disagreement (D) and predictive uncertainty (U) of the suppressed stream are combined and ranked against clean validation data: its reliability collapses, its fusion weight closes and a low-confidence flag is raised. The same frozen checkpoint now gives the defended decision.",
      route: "/attack",
    },
    {
      title: "A stealthier attacker",
      body: "A frozen-sensor attack (sample-and-hold) is harder to see than zeros. The evidence channels respond differently, and the Results page measures this as the detectability matrix.",
      route: "/attack",
      apply: () => {
        base();
        setAttack({ type: "mask", modality: DEMO.modality, severity: 1, fill: "hold", placement: "center" });
      },
    },
    {
      title: "Bursts in a live stream",
      body: "In deployment, suppression arrives as a burst across consecutive windows. The Stream Simulator replays the whole session. Drag the burst to see where fixed fusion fails, and where the defense flags and recovers.",
      route: "/stream",
      apply: () => base(),
    },
    {
      title: "Measured over 15 people",
      body: "All numbers come from the full LOSO evaluation with subject-bootstrap confidence intervals. The four pre-registered hypotheses (attack effectiveness, modality sensitivity, mitigation, detectability) are tested and reported as supported or not, without cherry-picking. That includes the price of the defense: its false alarms on clean windows of people it has never seen.",
      route: "/results",
      apply: () => setTask(DEMO.task),
    },
    {
      title: "Reproducible, live",
      body: "Every experiment can be relaunched from the browser. A quick run trains, calibrates and attacks a held-out subject in a few minutes, inside a 1.5 GB Docker budget.",
      route: "/experiments",
    },
  ];

  const open = demoStep != null;
  const step = open ? steps[Math.min(demoStep, steps.length - 1)] : null;

  useEffect(() => {
    if (!open || !step) return;
    step.apply?.();
    navigate(step.route);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demoStep]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") setDemoStep(Math.min(steps.length - 1, (demoStep ?? 0) + 1));
      if (e.key === "ArrowLeft") setDemoStep(Math.max(0, (demoStep ?? 0) - 1));
      if (e.key === "Escape") setDemoStep(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, demoStep, setDemoStep, steps.length]);

  return (
    <AnimatePresence>
      {open && step && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 16 }}
          transition={{ duration: 0.2 }}
          className="fixed bottom-6 right-6 z-50 w-[400px] overflow-hidden rounded-2xl border border-line bg-surface shadow-pop"
          role="dialog"
          aria-label="Guided demo"
        >
          <div className="h-1 bg-line">
            <div className="h-full bg-ink transition-all" style={{ width: `${((demoStep! + 1) / steps.length) * 100}%` }} />
          </div>
          <div className="p-5">
            <div className="mb-1 flex items-center justify-between">
              <span className="eyebrow">
                Guided demo · {demoStep! + 1} / {steps.length}
              </span>
              <button className="text-faint hover:text-ink" onClick={() => setDemoStep(null)} aria-label="Exit demo">
                <X className="h-4 w-4" />
              </button>
            </div>
            <h3 className="text-[16px] font-semibold text-ink">{step.title}</h3>
            <p className="mt-1.5 text-[13px] leading-relaxed text-ink-2">{step.body}</p>
            <div className="mt-4 flex items-center justify-between">
              <span className="flex items-center gap-1 text-[11px] text-muted">
                <Kbd>←</Kbd>
                <Kbd>→</Kbd>
                <Kbd>Esc</Kbd>
              </span>
              <div className="flex gap-2">
                <Button size="sm" variant="ghost" disabled={demoStep === 0} icon={<ArrowLeft className="h-3.5 w-3.5" />} onClick={() => setDemoStep(demoStep! - 1)}>
                  Back
                </Button>
                {demoStep! < steps.length - 1 ? (
                  <Button size="sm" variant="primary" onClick={() => setDemoStep(demoStep! + 1)}>
                    Next <ArrowRight className="h-3.5 w-3.5" />
                  </Button>
                ) : (
                  <Button size="sm" variant="primary" onClick={() => setDemoStep(null)}>
                    Finish
                  </Button>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
