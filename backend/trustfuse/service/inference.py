"""Inference service behind the interactive pages (Attack Lab, Stream Simulator, Data Explorer).

Every subject is served by the fold model that held that subject out (LOSO), so all interactive
predictions are made on data the model never saw during training.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

import numpy as np

from ..attacks.simulator import AttackSimulator, AttackSpec
from ..data.loader import DatasetLoader
from ..data.preprocess import Preprocessor, load_stream
from ..models.checkpoint import find_checkpoint
from ..reliability.engine import DEFAULT_CALIBRATION, DEFAULT_COMPONENTS, Decision, ModalityOutput
from ..specs import DATASETS, LABEL_HZ, MODALITY_SPECS, TASKS, get_task
from .bundle import BundleCache, FoldBundle

ADAPTIVE_FILLS = ("zero", "missing", "hold")
ADAPTIVE_SEVERITIES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _r(x, nd: int = 4):
    if isinstance(x, np.ndarray):
        return np.round(x.astype(float), nd).tolist()
    return round(float(x), nd)


def _decimate(x: np.ndarray, points: int) -> tuple[np.ndarray, np.ndarray]:
    """Min/max decimation of (C, N) -> (C, 2*points), preserving peaks for plotting."""
    n = x.shape[-1]
    if n <= 2 * points:
        return np.arange(n), x
    edges = np.linspace(0, n, points + 1).astype(int)
    idx, cols = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = x[:, a:b]
        lo, hi = a + seg.argmin(axis=1), a + seg.argmax(axis=1)
        pick = np.sort(np.stack([lo[0], hi[0]]))
        idx.extend(pick.tolist())
    idx = np.asarray(idx)
    return idx, x[:, idx]


class InferenceService:
    def __init__(self, capacity: int = 6):
        self.bundles = BundleCache(capacity)
        self._clean: OrderedDict[tuple, tuple[dict, Decision]] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ catalog
    def tasks(self) -> list[dict]:
        out = []
        for t in TASKS.values():
            loader = DatasetLoader(t.dataset)
            subjects = loader.available_subjects()
            models = sum(1 for s in DATASETS[t.dataset].subjects if find_checkpoint(t.key, "fusion", 0, s))
            out.append({"key": t.key, "title": t.title, "dataset": t.dataset, "dataset_name": DATASETS[t.dataset].name,
                        "modalities": list(t.modalities), "classes": list(t.classes), "window_s": t.window_s,
                        "stride_s": t.stride_s, "model_hz": t.model_hz, "headline": t.headline,
                        "subjects_ready": len(subjects), "subjects_total": len(DATASETS[t.dataset].subjects),
                        "fold_models": models})
        return out

    def subjects(self, task_key: str) -> list[dict]:
        task = get_task(task_key)
        out = []
        for s in DatasetLoader(task.dataset).available_subjects():
            st = load_stream(task_key, s)
            counts = np.bincount(st.windows.y, minlength=task.n_classes)
            out.append({"subject": s, "duration_s": st.duration_s, "windows": len(st.windows),
                        "class_windows": counts.tolist(),
                        "has_model": find_checkpoint(task_key, "fusion", 0, s) is not None})
        return out

    def bundle(self, task: str, subject: str, variant: str = "fusion", seed: int = 0) -> FoldBundle:
        return self.bundles.get(task, variant, seed, subject)

    # ------------------------------------------------------------------ signals
    def signals(self, task_key: str, subject: str, t0: float | None, t1: float | None, points: int = 1500) -> dict:
        task = get_task(task_key)
        st = load_stream(task_key, subject)
        t0 = max(0.0, t0 or 0.0)
        t1 = min(st.duration_s, t1 or st.duration_s)
        mods = {}
        for m in task.modalities:
            hz = task.model_hz[m]
            a, b = int(t0 * hz), int(t1 * hz)
            idx, x = _decimate(st.signals[m][:, a:b], points)
            mods[m] = {"t": _r((a + idx) / hz, 3), "values": [_r(c, 4) for c in x],
                       "channels": list(MODALITY_SPECS[m].channel_names), "unit": MODALITY_SPECS[m].unit,
                       "label": MODALITY_SPECS[m].label, "hz": hz}
        # label segments (raw dataset labels) inside the range
        lab = st.raw_label
        segs = []
        change = np.flatnonzero(np.diff(lab)) + 1
        bounds = np.concatenate([[0], change, [len(lab)]])
        names = DATASETS[task.dataset].label_names
        for a, b in zip(bounds[:-1], bounds[1:]):
            s0, s1 = a / LABEL_HZ, b / LABEL_HZ
            if s1 < t0 or s0 > t1:
                continue
            raw = int(lab[a])
            cls = task.label_map.get(raw)
            segs.append({"start": s0, "end": s1, "raw": raw, "name": names.get(raw, str(raw)),
                         "class": cls})
        return {"subject": subject, "t0": t0, "t1": t1, "duration_s": st.duration_s, "modalities": mods,
                "segments": segs, "classes": list(task.classes)}

    def windows(self, task_key: str, subject: str) -> dict:
        task = get_task(task_key)
        st = load_stream(task_key, subject)
        w = st.windows
        return {"subject": subject, "window_s": task.window_s, "stride_s": task.stride_s,
                "start_s": w.start_s.tolist(), "y": w.y.tolist(), "segment": w.segment.tolist(),
                "classes": list(task.classes)}

    # ------------------------------------------------------------------ helpers
    def _window_phys(self, task_key: str, subject: str, index: int) -> tuple[dict, int, float]:
        task = get_task(task_key)
        st = load_stream(task_key, subject)
        if not 0 <= index < len(st.windows):
            raise IndexError(f"window {index} out of range (0..{len(st.windows) - 1})")
        start = st.windows.start_s[index:index + 1]
        phys = Preprocessor(task).extract(st.signals, start)
        return phys, int(st.windows.y[index]), float(start[0])

    @staticmethod
    def _spec(attack: dict | None) -> AttackSpec:
        attack = dict(attack or {})
        return AttackSpec(**{k: v for k, v in attack.items() if k in AttackSpec.__dataclass_fields__})

    @staticmethod
    def _decision_payload(dec: Decision, i: int = 0) -> dict:
        def head(p):
            return {"probs": _r(p[i]), "pred": int(p[i].argmax()), "conf": _r(p[i].max())}
        return {"fixed": head(dec.fixed_probs), "defended": head(dec.defended_probs)}

    def _modality_payload(self, b: FoldBundle, dec: Decision, outs: dict[str, ModalityOutput], i: int = 0) -> dict:
        eng = b.engine
        dist = eng._distribution(dec.components, dec.calibration)
        out = {}
        for m in b.task.modalities:
            ev = dec.evidence[m]
            q = eng.quality[dec.calibration][m]
            out[m] = {
                "Q": _r(ev["Q"][i]), "D": _r(ev["D"][i]), "U": _r(ev["U"][i]), "zD": _r(ev["zD"][i]), "zU": _r(ev["zU"][i]),
                "anomaly": _r(dec.anomaly[m][i]), "R": _r(dec.reliability[m][i]), "w": _r(dec.weights[m][i]),
                "tau": _r(dist[m]["tau"]), "median": _r(dist[m]["median"]), "span": _r(dist[m]["span"]),
                "unimodal": _r(dec.aux_probs[m][i]),
                "quality": {"names": list(q.names), "values": _r(outs[m].q_feat[i]), "z": _r(outs[m].quality(dec.calibration)[1][i]),
                            "reference_median": _r(q.median), "reference_scale": _r(q.scale)},
            }
        return out

    # ------------------------------------------------------------------ attack lab
    def infer(self, task_key: str, subject: str, index: int, attack: dict | None = None, variant: str = "fusion",
              seed: int = 0, components: list[str] | None = None, strategy: str = "embedding_gate",
              calibration: str | None = None) -> dict:
        trace = []
        t = time.perf_counter()

        def step(label: str):
            nonlocal t
            now = time.perf_counter()
            trace.append({"step": len(trace) + 1, "label": label, "ms": round((now - t) * 1000, 2)})
            t = now

        b = self.bundle(task_key, subject, variant, seed)
        eng = b.engine
        comps = tuple(components) if components else DEFAULT_COMPONENTS
        calib = calibration or DEFAULT_CALIBRATION
        t = time.perf_counter()   # time the pipeline itself, not the one-off model load
        phys, y, start = self._window_phys(task_key, subject, index)
        step("Send synchronized window")
        clean_outs = eng.outputs(phys)
        clean = eng.decide(clean_outs, components=comps, strategy=strategy, calibration=calib)
        step("Forward clean signals")
        spec = self._spec(attack)
        attacked_phys = phys
        if spec.type in ("mask", "attenuate", "noise"):
            attacked_phys = AttackSimulator(b.task.modalities).apply(phys, spec, rng=np.random.default_rng(spec.seed))
        elif spec.type != "none":
            raise ValueError("window-level attack expected (mask, attenuate, noise or none)")
        step("Suppress selected modality")
        outs = dict(clean_outs)
        if spec.type != "none":
            outs[spec.modality] = eng.modality_output(spec.modality, attacked_phys[spec.modality])
        dec = eng.decide(outs, components=comps, strategy=strategy, calibration=calib)
        step("Generate baseline prediction")
        step("Compute reliability / disagreement")
        step("Down-weight unreliable modality")
        payload = self._decision_payload(dec)
        step("Generate final prediction")
        task = b.task
        signals = {}
        for m in task.modalities:
            signals[m] = {"hz": task.model_hz[m], "unit": MODALITY_SPECS[m].unit, "label": MODALITY_SPECS[m].label,
                          "channels": list(MODALITY_SPECS[m].channel_names),
                          "original": [_r(c, 4) for c in phys[m][0]],
                          "attacked": [[None if not np.isfinite(v) else round(float(v), 4) for v in c]
                                       for c in attacked_phys[m][0]]}
        clean_payload = self._decision_payload(clean)
        result = {
            "window": {"index": index, "start_s": start, "end_s": start + task.window_s, "label": y,
                       "label_name": task.classes[y]},
            "classes": list(task.classes), "attack": spec.to_dict(), "attack_label": spec.describe(),
            "signals": signals, **payload, "clean": clean_payload,
            "changed": {"fixed": payload["fixed"]["pred"] != clean_payload["fixed"]["pred"],
                        "defended": payload["defended"]["pred"] != clean_payload["defended"]["pred"]},
            "correct": {"fixed": payload["fixed"]["pred"] == y, "defended": payload["defended"]["pred"] == y},
            "modalities": self._modality_payload(b, dec, outs),
            "flag": bool(dec.flag[0]), "min_r": _r(dec.min_reliability()[0]),
            "flag_threshold": eng.alpha / len(task.modalities), "alpha": eng.alpha,
            "components": list(comps), "calibration": calib, "strategy": strategy, "model": b.meta(),
        }
        step("Log attack + performance")
        result["trace"] = trace
        return result

    def worst_modality(self, task_key: str, subject: str, index: int, fill: str = "zero", variant: str = "fusion",
                       seed: int = 0) -> dict:
        """Targeted (label-free, query-based) and oracle (label-aware) attacker choices for one window."""
        b = self.bundle(task_key, subject, variant, seed)
        eng = b.engine
        phys, y, _ = self._window_phys(task_key, subject, index)
        clean_outs = eng.outputs(phys)
        clean = eng.decide(clean_outs)
        cand = []
        sim = AttackSimulator(b.task.modalities)
        for m in b.task.modalities:
            att = sim.drop_modality(phys, m, fill)
            outs = dict(clean_outs)
            outs[m] = eng.modality_output(m, att[m])
            d = eng.decide(outs)
            cand.append({"modality": m, "tv": _r(0.5 * np.abs(d.fixed_probs[0] - clean.fixed_probs[0]).sum()),
                         "p_true_fixed": _r(d.fixed_probs[0, y]), "p_true_defended": _r(d.defended_probs[0, y]),
                         **self._decision_payload(d), "flag": bool(d.flag[0]), "R": _r(d.reliability[m][0]),
                         "w": _r(d.weights[m][0])})
        targeted = max(cand, key=lambda c: c["tv"])["modality"]
        oracle = min(cand, key=lambda c: c["p_true_fixed"])["modality"]
        return {"label": y, "clean": self._decision_payload(clean), "candidates": cand, "targeted": targeted,
                "oracle": oracle, "fill": fill}

    def adaptive(self, task_key: str, subject: str, index: int, variant: str = "fusion", seed: int = 0) -> dict:
        """Defense-aware attacker search for one window (smallest suppression that flips the defended
        decision without raising the flag)."""
        b = self.bundle(task_key, subject, variant, seed)
        eng = b.engine
        phys, y, _ = self._window_phys(task_key, subject, index)
        clean_outs = eng.outputs(phys)
        clean = eng.decide(clean_outs)
        cdp = int(clean.defended_pred[0])
        sim = AttackSimulator(b.task.modalities)
        cands = []
        for m in b.task.modalities:
            specs = [AttackSpec("mask", m, s, f, placement="center") for f in ADAPTIVE_FILLS for s in ADAPTIVE_SEVERITIES]
            specs += [AttackSpec("attenuate", m, s) for s in ADAPTIVE_SEVERITIES]
            for spec in specs:
                att = sim.apply(phys, spec, rng=np.random.default_rng(0))
                outs = dict(clean_outs)
                outs[m] = eng.modality_output(m, att[m])
                d = eng.decide(outs)
                cands.append({"modality": m, "type": spec.type, "fill": spec.fill if spec.type == "mask" else None,
                              "severity": spec.severity, "defended_pred": int(d.defended_pred[0]),
                              "fixed_pred": int(d.fixed_pred[0]), "flag": bool(d.flag[0]),
                              "suspicion": _r(d.suspicion()[0]), "R": _r(d.reliability[m][0]),
                              "p_clean_def": _r(d.defended_probs[0, cdp])})
        success = [c for c in cands if c["defended_pred"] != cdp and not c["flag"]]
        chosen = min(success, key=lambda c: (c["severity"], c["p_clean_def"])) if success else None
        flips_fixed = [c for c in cands if c["fixed_pred"] != int(clean.fixed_pred[0])]
        return {"label": y, "clean": self._decision_payload(clean), "candidates": cands, "chosen": chosen,
                "evasive_success": chosen is not None, "n_candidates": len(cands),
                "fixed_flippable": len(flips_fixed), "flagged": sum(c["flag"] for c in cands)}

    # ------------------------------------------------------------------ stream simulator
    def _clean_session(self, b: FoldBundle) -> tuple[dict, Decision, object]:
        key = (b.task.key, b.variant, b.seed, b.subject)
        with self._lock:
            if key in self._clean:
                self._clean.move_to_end(key)
                return self._clean[key]
        st = load_stream(b.task.key, b.subject)
        phys = Preprocessor(b.task).extract(st.signals, st.windows.start_s)
        outs = b.engine.outputs(phys)
        dec = b.engine.decide(outs)
        with self._lock:
            self._clean[key] = (outs, dec, st)
            while len(self._clean) > 4:
                self._clean.popitem(last=False)
        return outs, dec, st

    def timeline(self, task_key: str, subject: str, burst: dict | None = None, variant: str = "fusion",
                 seed: int = 0) -> dict:
        b = self.bundle(task_key, subject, variant, seed)
        task = b.task
        clean_outs, clean, st = self._clean_session(b)
        starts = st.windows.start_s
        W = task.window_s
        dec = clean
        affected = np.zeros(len(starts), bool)
        interval = None
        if burst and burst.get("modality"):
            m = burst["modality"]
            i0 = int(burst.get("start_index", 0))
            B = max(1, int(burst.get("length", 1)))
            i1 = min(len(starts) - 1, i0 + B - 1)
            begin, end = float(starts[i0]), float(starts[i1] + W)
            interval = {"start_s": begin, "end_s": end, "modality": m, "length": B, "fill": burst.get("fill", "zero")}
            aff = np.flatnonzero((starts < end) & (starts + W > begin))
            affected[aff] = True
            sig = AttackSimulator.burst_drop(st.signals, m, begin, end - begin, task.model_hz[m], burst.get("fill", "zero"))
            phys_m = Preprocessor(task).extract(sig, starts[aff], (m,))[m]
            outs = {j: clean_outs[j].take(aff) for j in task.modalities}
            outs[m] = b.engine.modality_output(m, phys_m)
            part = b.engine.decide(outs)
            dec = _patch(clean, part, aff)
        y = st.windows.y
        return {
            "subject": subject, "classes": list(task.classes), "window_s": W, "stride_s": task.stride_s,
            "start_s": starts.tolist(), "y": y.tolist(), "segment": st.windows.segment.tolist(),
            "fixed_pred": dec.fixed_pred.tolist(), "defended_pred": dec.defended_pred.tolist(),
            "fixed_conf": _r(dec.fixed_probs.max(1), 3), "defended_conf": _r(dec.defended_probs.max(1), 3),
            "clean_fixed_pred": clean.fixed_pred.tolist(), "clean_defended_pred": clean.defended_pred.tolist(),
            "reliability": {m: _r(v, 4) for m, v in dec.reliability.items()},
            "weights": {m: _r(v, 3) for m, v in dec.weights.items()},
            "flag": dec.flag.tolist(), "affected": affected.tolist(), "burst": interval,
            "summary": {
                "fixed_acc": float(np.mean(dec.fixed_pred == y)), "defended_acc": float(np.mean(dec.defended_pred == y)),
                "clean_fixed_acc": float(np.mean(clean.fixed_pred == y)),
                "clean_defended_acc": float(np.mean(clean.defended_pred == y)),
                "affected": int(affected.sum()),
                "fixed_errors_affected": int(np.sum((dec.fixed_pred != y) & affected)),
                "defended_errors_affected": int(np.sum((dec.defended_pred != y) & affected)),
                "flagged_affected": int(np.sum(dec.flag & affected)),
                "flagged_clean": int(np.sum(dec.flag & ~affected)),
            },
        }

    def calibration(self, task_key: str, subject: str, variant: str = "fusion", seed: int = 0,
                    components: list[str] | None = None, calibration: str | None = None) -> dict:
        b = self.bundle(task_key, subject, variant, seed)
        comps = tuple(components) if components else DEFAULT_COMPONENTS
        calib = calibration or DEFAULT_CALIBRATION
        return {"subject": subject, "components": list(comps), "calibration": calib, "alpha": b.engine.alpha,
                "flag_threshold": b.engine.alpha / len(b.task.modalities),
                "modalities": b.engine.calibration_summary(comps, calib), "model": b.meta()}


def _patch(clean: Decision, part: Decision, idx: np.ndarray) -> Decision:
    def put(a, p):
        a = a.copy()
        a[idx] = p
        return a
    return Decision(
        fixed_probs=put(clean.fixed_probs, part.fixed_probs), defended_probs=put(clean.defended_probs, part.defended_probs),
        aux_probs={m: put(clean.aux_probs[m], part.aux_probs[m]) for m in clean.aux_probs},
        evidence={m: {k: put(clean.evidence[m][k], part.evidence[m][k]) for k in clean.evidence[m]} for m in clean.evidence},
        anomaly={m: put(clean.anomaly[m], part.anomaly[m]) for m in clean.anomaly},
        reliability={m: put(clean.reliability[m], part.reliability[m]) for m in clean.reliability},
        weights={m: put(clean.weights[m], part.weights[m]) for m in clean.weights},
        flag=put(clean.flag, part.flag), components=clean.components,
        margin={m: put(clean.margin[m], part.margin[m]) for m in clean.margin}, calibration=clean.calibration)
