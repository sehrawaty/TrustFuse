import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { useApp } from "../lib/store";

/** Result set shown for the current task: the user's pick, else the canonical run for the task. */
export function useRun() {
  const { task, run, setRun } = useApp();
  const results = useQuery({ queryKey: ["results"], queryFn: api.results, refetchInterval: 15000 });
  const forTask = (results.data ?? []).filter((r) => r.task === task);
  const chosen = run[task] && forTask.some((r) => r.run_id === run[task]) ? run[task] : (forTask.find((r) => r.canonical) ?? forTask[0])?.run_id;
  return { runId: chosen, runs: forTask, setRun: (id: string) => setRun(task, id), loading: results.isLoading };
}

export function useSummary(runId?: string) {
  return useQuery({ queryKey: ["summary", runId], queryFn: () => api.summary(runId!), enabled: !!runId });
}

export function useConditions(runId?: string) {
  return useQuery({ queryKey: ["conditions", runId], queryFn: () => api.conditions(runId!), enabled: !!runId });
}

export function useDetection(runId?: string) {
  return useQuery({ queryKey: ["detection", runId], queryFn: () => api.detection(runId!), enabled: !!runId });
}
