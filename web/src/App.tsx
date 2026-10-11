import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";
import GuidedDemo from "./components/GuidedDemo";
import Shell from "./components/layout/Shell";
import { Skeleton } from "./components/ui";

const Overview = lazy(() => import("./pages/Overview"));
const DataExplorer = lazy(() => import("./pages/DataExplorer"));
const AttackLab = lazy(() => import("./pages/AttackLab"));
const StreamSimulator = lazy(() => import("./pages/StreamSimulator"));
const ReliabilityInspector = lazy(() => import("./pages/ReliabilityInspector"));
const Experiments = lazy(() => import("./pages/Experiments"));
const Results = lazy(() => import("./pages/Results"));
const Reports = lazy(() => import("./pages/Reports"));

function Loading() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-8 w-72" />
      <Skeleton className="h-4 w-[520px]" />
      <div className="grid grid-cols-3 gap-4 pt-4">
        <Skeleton className="h-40" />
        <Skeleton className="h-40" />
        <Skeleton className="h-40" />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <Shell>
      <Suspense fallback={<Loading />}>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/data" element={<DataExplorer />} />
          <Route path="/attack" element={<AttackLab />} />
          <Route path="/stream" element={<StreamSimulator />} />
          <Route path="/reliability" element={<ReliabilityInspector />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/results" element={<Results />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="*" element={<Overview />} />
        </Routes>
      </Suspense>
      <GuidedDemo />
    </Shell>
  );
}
