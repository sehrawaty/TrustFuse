"""Evaluator: runs the suppression matrix on one frozen fold model and its held-out subject.

Fixed fusion and reliability-aware fusion always receive the *identical* attacked windows
(matched comparison). Because an attack changes a single modality, only that modality's
encoder output is recomputed; the cached clean outputs of the other modalities are reused.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..attacks.simulator import AttackSimulator, AttackSpec
from ..data.preprocess import Preprocessor, load_stream
from ..data.windows import gather, normalize
from ..experiments.config import ExperimentConfig
from ..models.train import predict_logits
from ..reliability.engine import Decision, ModalityOutput
from ..service.bundle import FoldBundle
from .metrics import accuracy, auroc, balanced_accuracy, expected_calibration_error, macro_f1

CONF_HIGH = 0.8
# columns of the burst traceability table (one row per burst-affected window)
BURST_FIELDS = ("start_s", "y", "modality", "burst_windows", "placement", "burst_start_s", "burst_end_s",
                "fully_suppressed", "fixed_pred", "def_pred", "fixed_conf", "def_conf", "r_target", "w_target",
                "flag", "clean_fixed_pred", "clean_def_pred")


@dataclass
class FoldEvaluation:
    rows: list[dict]
    manifest: dict[str, np.ndarray] = field(default_factory=dict)
    windows: dict[str, np.ndarray] = field(default_factory=dict)
    scores: dict[str, np.ndarray] = field(default_factory=dict)
    bursts: dict[str, np.ndarray] = field(default_factory=dict)   # one record per burst-affected window


def _cond_id(**kw) -> str:
    parts = [kw.get("family", "")]
    for k in ("modality", "fill", "severity", "snr_db", "burst", "components", "strategy", "calibration"):
        v = kw.get(k)
        if v is not None and v != "":
            parts.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
    return "|".join(parts)


def select_decision(decs: list[Decision], choice: np.ndarray) -> Decision:
    """Per-window selection among candidate decisions (used by attacker strategies)."""
    rows = np.arange(choice.shape[0])

    def pick(arrs):
        return np.stack(arrs)[choice, rows]

    first = decs[0]
    mods = list(first.reliability)
    return Decision(
        fixed_probs=pick([d.fixed_probs for d in decs]),
        defended_probs=pick([d.defended_probs for d in decs]),
        aux_probs={m: pick([d.aux_probs[m] for d in decs]) for m in mods},
        evidence={m: {k: pick([d.evidence[m][k] for d in decs]) for k in first.evidence[m]} for m in mods},
        anomaly={m: pick([d.anomaly[m] for d in decs]) for m in mods},
        reliability={m: pick([d.reliability[m] for d in decs]) for m in mods},
        weights={m: pick([d.weights[m] for d in decs]) for m in mods},
        flag=pick([d.flag for d in decs]),
        components=first.components,
        margin={m: pick([d.margin[m] for d in decs]) for m in mods},
        calibration=first.calibration,
    )


def component_scores(dec: Decision) -> dict[str, np.ndarray]:
    """Per-evidence suspicion (max over modalities) for the detectability matrix."""
    mods = list(dec.evidence)
    return {"Q": np.max([dec.evidence[m]["Q"] for m in mods], axis=0),
            "D": np.max([dec.evidence[m]["zD"] for m in mods], axis=0),
            "U": np.max([dec.evidence[m]["zU"] for m in mods], axis=0),
            "R": dec.suspicion()}


def compute_metrics(y: np.ndarray, dec: Decision, clean: Decision, k: int, target: str | None,
                    idx: np.ndarray | None = None) -> dict:
    if idx is not None:
        y = y[idx]
        clean = _subset(clean, idx)
    fp, dp = dec.fixed_pred, dec.defended_pred
    cfp, cdp = clean.fixed_pred, clean.defended_pred
    fconf, dconf = dec.fixed_probs.max(1), dec.defended_probs.max(1)
    rows = np.arange(len(y))
    unflag = ~dec.flag
    f_ok, d_ok = cfp == y, cdp == y
    out = {
        "n": int(len(y)),
        "fixed_acc": accuracy(y, fp), "fixed_f1": macro_f1(y, fp, k), "fixed_bacc": balanced_accuracy(y, fp, k),
        "def_acc": accuracy(y, dp), "def_f1": macro_f1(y, dp, k), "def_bacc": balanced_accuracy(y, dp, k),
        "fixed_flip": float(np.mean(fp != cfp)), "def_flip": float(np.mean(dp != cdp)),
        "fixed_asr": float(np.mean(fp[f_ok] != y[f_ok])) if f_ok.any() else float("nan"),
        "def_asr": float(np.mean(dp[d_ok] != y[d_ok])) if d_ok.any() else float("nan"),
        "fixed_conf": float(fconf.mean()), "def_conf": float(dconf.mean()),
        "fixed_ptrue": float(dec.fixed_probs[rows, y].mean()), "def_ptrue": float(dec.defended_probs[rows, y].mean()),
        "fixed_cwrong": float(np.mean((fp != y) & (fconf >= CONF_HIGH))),
        "def_cwrong": float(np.mean((dp != y) & (dconf >= CONF_HIGH))),
        "fixed_ece": expected_calibration_error(fconf, fp == y), "def_ece": expected_calibration_error(dconf, dp == y),
        "flag_rate": float(dec.flag.mean()),
        "sel_acc": accuracy(y[unflag], dp[unflag]) if unflag.any() else float("nan"),
        "coverage": float(unflag.mean()),
        "evasive_def_flip": float(np.mean((dp != cdp) & unflag)),
        "min_r": float(np.median(dec.min_reliability())),
    }
    if target is not None:
        out["r_target"] = float(np.median(dec.reliability[target]))
        out["w_target"] = float(np.mean(dec.weights[target]))
    return out


def _subset(dec: Decision, idx: np.ndarray) -> Decision:
    return Decision(
        fixed_probs=dec.fixed_probs[idx], defended_probs=dec.defended_probs[idx],
        aux_probs={m: v[idx] for m, v in dec.aux_probs.items()},
        evidence={m: {k: v[idx] for k, v in e.items()} for m, e in dec.evidence.items()},
        anomaly={m: v[idx] for m, v in dec.anomaly.items()},
        reliability={m: v[idx] for m, v in dec.reliability.items()},
        weights={m: v[idx] for m, v in dec.weights.items()}, flag=dec.flag[idx], components=dec.components,
        margin={m: v[idx] for m, v in dec.margin.items()}, calibration=dec.calibration)


class Evaluator:
    def __init__(self, bundle: FoldBundle, cfg: ExperimentConfig, log: Callable[[str], None] | None = None,
                 should_stop: Callable[[], bool] | None = None):
        self.b = bundle
        self.cfg = cfg
        self.task = bundle.task
        self.k = bundle.task.n_classes
        self.log = log or (lambda s: None)
        self.should_stop = should_stop or (lambda: False)
        self.ws = gather(self.task.key, [bundle.subject])
        self.y = self.ws.y
        self.base = {"task": self.task.key, "variant": bundle.variant, "seed": bundle.seed, "subject": bundle.subject}

    # -- unimodal baselines: clean performance only ----------------------------------------------
    def evaluate_unimodal(self) -> FoldEvaluation:
        x = normalize(self.task.key, {m: self.ws.phys[m] for m in self.b.model.modalities}, self.b.norm)
        pred = predict_logits(self.b.model, x).argmax(1)
        row = {**self.base, "cond": "clean", "family": "clean", "attack": "none", "modality": None,
               "n": int(len(self.y)), "fixed_acc": accuracy(self.y, pred), "fixed_f1": macro_f1(self.y, pred, self.k),
               "fixed_bacc": balanced_accuracy(self.y, pred, self.k)}
        return FoldEvaluation(rows=[row])

    # -- fusion models: full matrix ----------------------------------------------------------------
    def _seed(self, *parts) -> int:
        h = hashlib.sha256("|".join(map(str, (self.cfg.attacks.seed, self.b.subject) + parts)).encode())
        return int.from_bytes(h.digest()[:4], "little")

    def evaluate(self, with_manifest: bool = False) -> FoldEvaluation:
        eng = self.b.engine
        assert eng is not None
        A = self.cfg.attacks
        mods = self.task.modalities
        comps = tuple(self.cfg.reliability.components)
        strategy = self.cfg.reliability.strategy
        calib = self.cfg.reliability.calibration
        sim = AttackSimulator(mods)
        clean_outs = eng.outputs(self.ws.phys)
        clean = eng.decide(clean_outs, components=comps, strategy=strategy, calibration=calib)
        rows: list[dict] = []
        manifest: dict[str, np.ndarray] = {}
        scores: dict[str, np.ndarray] = {}
        cache: dict[tuple, ModalityOutput] = {}
        decs: dict[tuple, Decision] = {}

        def out_for(spec: AttackSpec) -> ModalityOutput:
            key = (spec.type, spec.modality, spec.fill, round(spec.severity, 4), spec.snr_db)
            if key not in cache:
                rng = np.random.default_rng(self._seed(*key))
                x = sim.apply({spec.modality: self.ws.phys[spec.modality]}, spec, rng=rng)[spec.modality]
                cache[key] = eng.modality_output(spec.modality, x)
            return cache[key]

        def dec_for(spec: AttackSpec, components=comps, strat=strategy, cal=calib) -> Decision:
            key = (spec.type, spec.modality, spec.fill, round(spec.severity, 4), spec.snr_db, components, strat, cal)
            if key not in decs:
                outs = dict(clean_outs)
                outs[spec.modality] = out_for(spec)
                decs[key] = eng.decide(outs, components=components, strategy=strat, calibration=cal)
            return decs[key]

        def record(family: str, dec: Decision, target: str | None, store_scores: bool = False,
                   manifest_target: np.ndarray | None = None, **meta) -> None:
            cid = _cond_id(family=family, modality=target, **meta)
            row = {**self.base, "cond": cid, "family": family, "modality": target, **meta}
            row.update(compute_metrics(self.y, dec, clean, self.k, target))
            if family != "clean":
                row["auroc"] = auroc(clean.suspicion(), dec.suspicion())
                cs, cc = component_scores(dec), component_scores(clean)
                for c in ("Q", "D", "U"):
                    row[f"auroc_{c}"] = auroc(cc[c], cs[c])
            rows.append(row)
            if store_scores:
                scores[cid] = dec.suspicion().astype(np.float32)
            if with_manifest:
                tgt = dec.reliability[target] if target else dec.min_reliability()
                w = dec.weights[target] if target else np.min(np.stack(list(dec.weights.values())), 0)
                manifest[cid] = np.stack([dec.fixed_pred, dec.defended_pred, dec.fixed_probs.max(1),
                                          dec.defended_probs.max(1), tgt, w, dec.flag.astype(float)]).astype(np.float16)
                if manifest_target is not None:
                    manifest[cid + "#target"] = manifest_target.astype(np.int8)

        record("clean", clean, None, store_scores=True, attack="none")

        # 1) window-level suppression matrix
        for m in mods:
            if self.should_stop():
                raise InterruptedError
            for fill in A.mask_fills:
                for sev in A.severities:
                    spec = AttackSpec("mask", m, sev, fill, placement=A.placement)
                    record("mask", dec_for(spec), m, store_scores=(sev == 1.0), attack="mask", fill=fill, severity=sev)
            for sev in A.attenuations:
                spec = AttackSpec("attenuate", m, sev)
                record("attenuate", dec_for(spec), m, store_scores=(sev in (0.6, 0.8)), attack="attenuate", severity=sev)
            for snr in A.noise_snr_db:
                spec = AttackSpec("noise", m, 1.0, snr_db=snr)
                record("noise", dec_for(spec), m, store_scores=(snr == 5.0), attack="noise", snr_db=snr)
        self.log(f"  {self.b.subject}: window-level matrix done ({len(rows)} conditions)")

        # 2) attacker strategies (one suppressed modality per window, full suppression)
        full = [dec_for(AttackSpec("mask", m, 1.0, A.strategy_fill, placement="start")) for m in mods]
        n = len(self.y)
        rows_idx = np.arange(n)
        if "random_drop" in A.strategies:
            choice = np.random.default_rng(self._seed("random")).integers(0, len(mods), size=n)
            record("random_drop", select_decision(full, choice), None, store_scores=True,
                   manifest_target=choice, attack="random_drop", fill=A.strategy_fill)
        if "targeted" in A.strategies:
            # label-free, query-based: suppress the modality that moves the fixed-fusion output most
            tv = np.stack([0.5 * np.abs(d.fixed_probs - clean.fixed_probs).sum(1) for d in full])
            choice = tv.argmax(0)
            record("targeted", select_decision(full, choice), None, store_scores=True,
                   manifest_target=choice, attack="targeted", fill=A.strategy_fill)
        if "oracle" in A.strategies:
            # label-aware worst case for fixed fusion
            ptrue = np.stack([d.fixed_probs[rows_idx, self.y] for d in full])
            choice = ptrue.argmin(0)
            record("oracle", select_decision(full, choice), None, store_scores=True,
                   manifest_target=choice, attack="oracle", fill=A.strategy_fill)
        if "adaptive" in A.strategies:
            cands: list[Decision] = []
            cost: list[float] = []
            for mi, m in enumerate(mods):
                for fill in ("zero", "missing", "hold"):
                    for sev in A.adaptive_severities:
                        cands.append(dec_for(AttackSpec("mask", m, sev, fill, placement=A.placement)))
                        cost.append(sev)
                for sev in A.adaptive_severities:
                    cands.append(dec_for(AttackSpec("attenuate", m, sev)))
                    cost.append(sev)
            choice = self._adaptive_choice(cands, np.asarray(cost), clean)
            per_mod = len(cands) // len(mods)
            record("adaptive", select_decision(cands, choice), None, store_scores=True,
                   manifest_target=choice // per_mod, attack="adaptive")
        self.log(f"  {self.b.subject}: attacker strategies done")

        # 3) stream-level bursts
        burst_rows, burst_records = self._bursts(eng, clean, clean_outs, comps, strategy, calib, with_manifest)
        rows += burst_rows
        self.log(f"  {self.b.subject}: bursts done")

        # 4) ablations: evidence subsets and the fusion strategy under the primary calibration,
        #    then the calibration extension (the other scheme) for selected evidence sets
        R = self.cfg.reliability
        abl_specs = [AttackSpec("none")]
        for m in mods:
            abl_specs += [AttackSpec("mask", m, 1.0, self.cfg.stats.primary_fill, placement="start"),
                          AttackSpec("attenuate", m, 0.8), AttackSpec("mask", m, 1.0, "hold", placement="start")]
        variants = [(tuple(c), strategy, calib) for c in R.ablations]
        if R.ablations:
            variants.append((comps, "poe" if strategy != "poe" else "embedding_gate", calib))
        other = "pooled" if calib == "validation" else "validation"
        variants += [(tuple(c), strategy, other) for c in R.calibration_ablations]
        for c, strat, cal in variants:
            clean_v = eng.decide(clean_outs, components=c, strategy=strat, calibration=cal)
            for spec in abl_specs:
                dec = clean_v if spec.type == "none" else dec_for(spec, components=c, strat=strat, cal=cal)
                cid = _cond_id(family="ablation", modality=spec.modality, fill=spec.fill if spec.type == "mask" else None,
                               severity=spec.severity if spec.type != "none" else None,
                               components="".join(c), strategy=strat, calibration=cal) + f"|attack={spec.type}"
                row = {**self.base, "cond": cid, "family": "ablation", "attack": spec.type, "modality": spec.modality,
                       "components": "".join(c), "strategy": strat, "calibration": cal,
                       "fill": spec.fill if spec.type == "mask" else None,
                       "severity": spec.severity if spec.type != "none" else None}
                row.update(compute_metrics(self.y, dec, clean_v, self.k, spec.modality))
                if spec.type != "none":
                    row["auroc"] = auroc(clean_v.suspicion(), dec.suspicion())
                rows.append(row)

        windows = {"start_s": self.ws.start_s.astype(np.int32), "y": self.y.astype(np.int8)}
        return FoldEvaluation(rows=rows, manifest=manifest, windows=windows, scores=scores, bursts=burst_records)

    @staticmethod
    def _adaptive_choice(cands: list[Decision], cost: np.ndarray, clean: Decision) -> np.ndarray:
        """Defense-aware attacker: flip the *defended* decision while keeping the flag down, at the
        lowest suppression cost. Without an evasive flip it keeps the decision that most reduces the
        defended confidence in the clean prediction among unflagged candidates; if every candidate
        is flagged it chooses the least suspicious one."""
        n = clean.flag.shape[0]
        rows = np.arange(n)
        cdp = clean.defended_pred
        flip = np.stack([c.defended_pred != cdp for c in cands])          # (C, n)
        flag = np.stack([c.flag for c in cands])
        p_clean_cls = np.stack([c.defended_probs[rows, cdp] for c in cands])
        susp = np.stack([c.suspicion() for c in cands])
        success = flip & ~flag
        big = 1e9
        succ_cost = np.where(success, cost[:, None] + 1e-3 * p_clean_cls, big)
        choice = succ_cost.argmin(0)
        none = ~success.any(0)
        if none.any():
            unflag_score = np.where(~flag, p_clean_cls, big)
            alt = unflag_score.argmin(0)
            all_flag = flag.all(0)
            alt = np.where(all_flag, susp.argmin(0), alt)
            choice = np.where(none, alt, choice)
        return choice

    def _bursts(self, eng, clean: Decision, clean_outs: dict, comps, strategy, calib: str,
                with_manifest: bool = False) -> tuple[list[dict], dict[str, np.ndarray]]:
        """Aggregated burst rows plus, for the manifest model, one traceability record per
        burst-affected window (target modality, burst length, placement, interval, decisions)."""
        A = self.cfg.attacks
        task = self.task
        stream = load_stream(task.key, self.b.subject)
        pp = Preprocessor(task)
        starts = self.ws.start_s
        seg = self.ws.segment
        W, S = task.window_s, task.stride_s
        out_rows = []
        rec: dict[str, list] = {k: [] for k in BURST_FIELDS}
        for B in A.burst_windows:
            valid = np.array([i for i in range(len(starts) - B + 1)
                              if seg[i] == seg[i + B - 1] and starts[i + B - 1] - starts[i] == (B - 1) * S])
            if valid.size == 0:
                continue
            rng = np.random.default_rng(self._seed("burst", B))
            picks = rng.choice(valid, size=min(A.burst_placements_per_subject, valid.size), replace=False)
            for m in task.modalities:
                hz = task.model_hz[m]
                full_idx, part_idx, decs_full, decs_part, latency = [], [], [], [], []
                for p, i in enumerate(picks):
                    begin, end = float(starts[i]), float(starts[i + B - 1] + W)
                    aff = np.flatnonzero((starts < end) & (starts + W > begin))
                    full = (starts[aff] >= begin) & (starts[aff] + W <= end)
                    sig = AttackSimulator.burst_drop(stream.signals, m, begin, end - begin, hz, A.burst_fill)
                    phys_m = pp.extract(sig, starts[aff], (m,))[m]
                    outs = {j: clean_outs[j].take(aff) for j in task.modalities}
                    outs[m] = eng.modality_output(m, phys_m)
                    dec = eng.decide(outs, components=comps, strategy=strategy, calibration=calib)
                    if with_manifest:
                        n_aff = len(aff)
                        rec["start_s"].append(starts[aff])
                        rec["y"].append(self.y[aff])
                        rec["modality"].append(np.full(n_aff, m))
                        rec["burst_windows"].append(np.full(n_aff, B))
                        rec["placement"].append(np.full(n_aff, p))
                        rec["burst_start_s"].append(np.full(n_aff, begin))
                        rec["burst_end_s"].append(np.full(n_aff, end))
                        rec["fully_suppressed"].append(full)
                        rec["fixed_pred"].append(dec.fixed_pred)
                        rec["def_pred"].append(dec.defended_pred)
                        rec["fixed_conf"].append(dec.fixed_probs.max(1))
                        rec["def_conf"].append(dec.defended_probs.max(1))
                        rec["r_target"].append(dec.reliability[m])
                        rec["w_target"].append(dec.weights[m])
                        rec["flag"].append(dec.flag)
                        rec["clean_fixed_pred"].append(clean.fixed_pred[aff])
                        rec["clean_def_pred"].append(clean.defended_pred[aff])
                    decs_full.append(_subset(dec, np.flatnonzero(full)))
                    decs_part.append(_subset(dec, np.flatnonzero(~full)))
                    full_idx.append(aff[full])
                    part_idx.append(aff[~full])
                    flagged = starts[aff][dec.flag]
                    latency.append(float(flagged.min() - begin) if flagged.size else float("nan"))
                for scope, idx_list, dl in (("full", full_idx, decs_full), ("partial", part_idx, decs_part)):
                    idx = np.concatenate(idx_list) if idx_list else np.zeros(0, int)
                    if idx.size == 0:
                        continue
                    dec = _concat(dl)
                    row = {**self.base, "cond": _cond_id(family="burst", modality=m, burst=B) + f"|scope={scope}",
                           "family": "burst", "attack": "burst", "modality": m, "burst": B, "scope": scope,
                           "fill": A.burst_fill, "placements": int(len(picks))}
                    row.update(compute_metrics(self.y, dec, clean, self.k, m, idx=idx))
                    row["latency_s"] = float(np.nanmean(latency)) if np.isfinite(latency).any() else float("nan")
                    row["detected_bursts"] = float(np.mean(np.isfinite(latency)))
                    out_rows.append(row)
        records = {k: np.concatenate(v) for k, v in rec.items()} if rec["start_s"] else {}
        return out_rows, records


def _concat(decs: list[Decision]) -> Decision:
    f = decs[0]
    cat = np.concatenate
    return Decision(
        fixed_probs=cat([d.fixed_probs for d in decs]), defended_probs=cat([d.defended_probs for d in decs]),
        aux_probs={m: cat([d.aux_probs[m] for d in decs]) for m in f.aux_probs},
        evidence={m: {k: cat([d.evidence[m][k] for d in decs]) for k in f.evidence[m]} for m in f.evidence},
        anomaly={m: cat([d.anomaly[m] for d in decs]) for m in f.anomaly},
        reliability={m: cat([d.reliability[m] for d in decs]) for m in f.reliability},
        weights={m: cat([d.weights[m] for d in decs]) for m in f.weights},
        flag=cat([d.flag for d in decs]), components=f.components,
        margin={m: cat([d.margin[m] for d in decs]) for m in f.margin}, calibration=f.calibration)
