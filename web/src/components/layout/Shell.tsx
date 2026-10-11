import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import {
  Activity,
  BarChart3,
  Database,
  FileDown,
  FlaskConical,
  Gauge,
  LayoutDashboard,
  Play,
  Radar,
  Waves,
} from "lucide-react";
import type { ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { api } from "../../lib/api";
import { bytes } from "../../lib/format";
import { useApp } from "../../lib/store";
import { Button, Kbd, Segmented, Tip } from "../ui";

const NAV: { group: string; items: { to: string; label: string; icon: ReactNode; hint: string }[] }[] = [
  { group: "", items: [{ to: "/", label: "Overview", icon: <LayoutDashboard />, hint: "Threat model, pipeline and headline findings" }] },
  { group: "Explore", items: [{ to: "/data", label: "Data Explorer", icon: <Database />, hint: "Synchronized wrist streams and labels" }] },
  {
    group: "Attack & defend",
    items: [
      { to: "/attack", label: "Attack Lab", icon: <Radar />, hint: "Suppress a modality and watch both fusion pipelines" },
      { to: "/stream", label: "Stream Simulator", icon: <Waves />, hint: "Burst suppression across a whole session" },
      { to: "/reliability", label: "Reliability Inspector", icon: <Gauge />, hint: "How Q, D and U become R and fusion weights" },
    ],
  },
  {
    group: "Evaluate",
    items: [
      { to: "/experiments", label: "Experiments", icon: <FlaskConical />, hint: "Configure and run experiments live" },
      { to: "/results", label: "Results", icon: <BarChart3 />, hint: "Security evaluation and hypothesis tests" },
      { to: "/reports", label: "Reports & Method", icon: <FileDown />, hint: "Paper figures, LaTeX tables, manifests" },
    ],
  },
];

export const TASK_SHORT: Record<string, string> = {
  wesad_binary: "WESAD · binary",
  wesad_3class: "WESAD · 3-class",
  dalia_activity: "PPG-DaLiA",
};

function Sidebar() {
  const sys = useQuery({ queryKey: ["system"], queryFn: api.system, refetchInterval: 5000 });
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 10000, retry: 0 });
  const mem = sys.data?.memory;
  // inside Docker the container cgroup is the meaningful number; outside it, the API process RSS
  const limit = mem?.cgroup_limit_bytes;
  const used = limit ? mem?.cgroup_used_bytes : mem?.rss_bytes;
  const active = sys.data?.jobs_active?.[0];
  return (
    <aside className="fixed inset-y-0 left-0 z-30 flex w-[232px] flex-col border-r border-line bg-surface">
      <div className="flex h-14 items-center gap-2.5 border-b border-line px-5">
        <svg viewBox="0 0 32 32" className="h-7 w-7" aria-hidden>
          <rect width="32" height="32" rx="8" fill="#101828" />
          <path d="M7 17h4l2.2-6 3.6 12 2.4-8 1.4 2H25" fill="none" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <div className="leading-tight">
          <div className="text-[14.5px] font-semibold tracking-tight">TrustFuse</div>
          <div className="text-[10.5px] text-muted">Modality-suppression security</div>
        </div>
      </div>
      <nav className="flex-1 overflow-y-auto px-3 py-4">
        {NAV.map((g) => (
          <div key={g.group || "root"} className="mb-4">
            {g.group && <div className="eyebrow mb-1.5 px-2.5">{g.group}</div>}
            {g.items.map((it) => (
              <Tip key={it.to} content={it.hint} side="right">
                {/* span wrapper: the tooltip Slot would otherwise stringify NavLink's className function */}
                <span className="block">
                <NavLink
                  to={it.to}
                  end={it.to === "/"}
                  className={({ isActive }) =>
                    clsx(
                      "mb-0.5 flex items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-[13px] transition-colors [&_svg]:h-4 [&_svg]:w-4",
                      isActive ? "bg-sunken font-medium text-ink ring-1 ring-line" : "text-ink-2 hover:bg-sunken hover:text-ink",
                    )
                  }
                >
                  {it.icon}
                  {it.label}
                </NavLink>
                </span>
              </Tip>
            ))}
          </div>
        ))}
      </nav>
      <div className="space-y-2.5 border-t border-line px-5 py-4 text-xs">
        {active && (
          <NavLink to="/experiments" className="block rounded-lg border border-line bg-sunken px-2.5 py-2">
            <div className="flex items-center gap-1.5 font-medium text-ink">
              <Activity className="h-3.5 w-3.5 animate-pulseDot text-defended" /> Job {active.status}
            </div>
            <div className="mt-1 truncate text-muted">{active.name}</div>
            <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-line">
              <div className="h-full bg-ink transition-all" style={{ width: `${Math.round(active.progress * 100)}%` }} />
            </div>
          </NavLink>
        )}
        <div className="flex items-center justify-between text-muted">
          <span className="flex items-center gap-1.5">
            <span className={clsx("h-1.5 w-1.5 rounded-full", health.isSuccess ? "bg-good" : "bg-attack")} />
            API {health.isSuccess ? "online" : "offline"}
          </span>
          <span className="num">v{health.data?.version ?? "—"}</span>
        </div>
        {used != null && (
          <div className="flex items-center justify-between text-muted">
            <span>{limit ? "Container memory" : "API memory"}</span>
            <span className="num">
              {bytes(used)}
              {limit ? ` / ${bytes(limit)}` : ""}
            </span>
          </div>
        )}
      </div>
    </aside>
  );
}

function TopBar() {
  const { task, setTask, setDemoStep } = useApp();
  const loc = useLocation();
  const taskAware = !["/experiments", "/reports"].includes(loc.pathname);
  return (
    <header className="sticky top-0 z-20 flex h-14 items-center justify-between gap-4 border-b border-line bg-paper/85 px-8 backdrop-blur">
      <div className="flex items-center gap-3">
        <span className="eyebrow">Task</span>
        <Segmented
          value={task}
          onChange={setTask}
          options={Object.entries(TASK_SHORT).map(([value, label]) => ({ value, label, disabled: !taskAware && false }))}
        />
      </div>
      <div className="flex items-center gap-3">
        <span className="hidden items-center gap-1 text-xs text-muted xl:flex">
          <Kbd>←</Kbd>
          <Kbd>→</Kbd> step demo
        </span>
        <Button variant="primary" icon={<Play className="h-3.5 w-3.5" />} onClick={() => setDemoStep(0)}>
          Guided demo
        </Button>
      </div>
    </header>
  );
}

export default function Shell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen">
      <Sidebar />
      <div className="pl-[232px]">
        <TopBar />
        <main className="mx-auto max-w-[1440px] px-8 pb-16 pt-7">{children}</main>
      </div>
    </div>
  );
}
