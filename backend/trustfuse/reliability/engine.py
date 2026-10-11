"""ReliabilityEngine: per-modality evidence -> reliability scores -> fusion weights.

Evidence for modality m of a window:
    Q_m  signal-quality anomaly       (model-free, see ``quality.py``)
    D_m  cross-modal disagreement     JS( g_m(h_m) || mean_{j != m} g_j(h_j) )   [bits]
    U_m  predictive uncertainty       entropy of the MC-dropout mean of g_m, normalised by log K

Calibration follows the project report (sections 3.1.2 and 4.3.3): Q, D and U are normalised
with clean *validation* data, combined into one anomaly score a_m = Q_m + z(D_m) + z(U_m), and
every scaling parameter and threshold is fitted on the clean validation subjects only and frozen
before any attack is evaluated. The validation subjects are disjoint from the training and test
subjects, so the model was not fitted on them. The reliability score

    R_m = (1 + #{ clean-validation windows with a >= a_m }) / (n + 1)

is a conformal p-value: for clean data exchangeable with the validation windows
P(R_m <= t) <= t. The low-confidence flag fires when min_m R_m <= alpha / M (Bonferroni over
modalities), which bounds its false-rejection rate on such data by alpha.

Two calibration schemes are available (``calibration``):

    validation  the report's scheme and the default: every reference comes from the validation
                subjects, for every evidence set.
    pooled      extension, kept for comparison: the model-free Q indicators are referenced to
                every non-test subject (training + validation), and the evidence set {Q} ranks
                against that pooled reference. Evidence sets containing the model-based D or U
                still rank against the validation subjects.

The fusion weight

    w_m = clip(1 - (a_m - tau_m) / (span * (tau_m - median_m)), 0, 1),   tau_m = conformal threshold

keeps clean-looking modalities at full weight and closes the gate gradually as evidence of
suppression grows. The defended representation is h~_m = w_m h_m + (1 - w_m) mu_m, with mu_m the
embedding of a neutral (training-mean) input, and goes through the *same frozen* classifier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from ..data.preprocess import NormStats, Preprocessor
from ..evaluation.metrics import entropy, js_divergence
from ..models.fusion import FusionModel
from ..specs import TaskSpec
from .quality import QualityCalibration, raw_features

COMPONENTS = ("Q", "D", "U")
CALIBRATIONS = ("validation", "pooled")
# Default operating point = the design fixed in the project report before any experiment:
# R combines Q, D and U, and all scaling parameters and thresholds come from clean validation data.
DEFAULT_COMPONENTS = COMPONENTS
DEFAULT_CALIBRATION = "validation"
Z_CLIP = (-3.0, 50.0)


@dataclass
class ModalityOutput:
    h: np.ndarray          # (B, E) deterministic embedding
    aux: np.ndarray        # (B, K) deterministic unimodal-head probabilities
    u: np.ndarray          # (B,)   normalised predictive entropy (MC dropout)
    q: np.ndarray          # (B,)   signal-quality anomaly (>= 0), validation-referenced
    q_feat: np.ndarray     # (B, F) raw quality indicators
    q_z: np.ndarray        # (B, F) directional z-scores of the indicators, validation-referenced
    q_pool: np.ndarray     # (B,)   the same anomaly referenced to every non-test subject (extension)
    q_z_pool: np.ndarray   # (B, F)

    def take(self, idx) -> "ModalityOutput":
        return ModalityOutput(self.h[idx], self.aux[idx], self.u[idx], self.q[idx], self.q_feat[idx], self.q_z[idx],
                              self.q_pool[idx], self.q_z_pool[idx])

    def quality(self, calibration: str) -> tuple[np.ndarray, np.ndarray]:
        """(Q anomaly, indicator z-scores) under the given calibration scheme."""
        return (self.q, self.q_z) if calibration == "validation" else (self.q_pool, self.q_z_pool)


@dataclass
class Decision:
    fixed_probs: np.ndarray                  # (B, K) standard fusion
    defended_probs: np.ndarray               # (B, K) reliability-aware fusion
    aux_probs: dict[str, np.ndarray]
    evidence: dict[str, dict[str, np.ndarray]]   # m -> {Q, D, U, zD, zU}
    anomaly: dict[str, np.ndarray]
    reliability: dict[str, np.ndarray]       # conformal p-values R_m
    weights: dict[str, np.ndarray]
    flag: np.ndarray                         # (B,) low-confidence / suspected suppression
    components: tuple[str, ...] = field(default=DEFAULT_COMPONENTS)
    margin: dict[str, np.ndarray] = field(default_factory=dict)   # (a_m - tau_m) / span_m
    calibration: str = DEFAULT_CALIBRATION

    @property
    def fixed_pred(self) -> np.ndarray:
        return self.fixed_probs.argmax(1)

    @property
    def defended_pred(self) -> np.ndarray:
        return self.defended_probs.argmax(1)

    def min_reliability(self) -> np.ndarray:
        return np.min(np.stack(list(self.reliability.values()), axis=1), axis=1)

    def suspicion(self) -> np.ndarray:
        """Detection score for ROC analysis: the largest calibrated anomaly margin
        (a_m - tau_m) / span_m over modalities. Unlike -log R it does not saturate at the
        conformal floor 1/(n+1), so strongly suppressed windows remain rankable."""
        return np.max(np.stack(list(self.margin.values()), axis=1), axis=1)


def _softmax(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=-1).numpy()


class ReliabilityEngine:
    def __init__(self, model: FusionModel, task: TaskSpec, norm: NormStats, mc_samples: int = 8,
                 alpha: float = 0.05, gate_span: float = 1.0, mc_seed: int = 0):
        if not isinstance(model, FusionModel):
            raise TypeError("the reliability engine needs a FusionModel with auxiliary heads")
        self.model = model.eval()
        self.task = task
        self.norm = norm
        self.modalities = tuple(model.modalities)
        self.mc_samples = mc_samples
        self.alpha = alpha
        self.gate_span = gate_span
        self.mc_seed = mc_seed
        self.pp = Preprocessor(task)
        self.calibration = DEFAULT_CALIBRATION         # scheme used when a call does not name one
        self.quality: dict[str, dict[str, QualityCalibration]] = {}   # scheme -> modality -> Q reference
        self.mu: dict[str, np.ndarray] = {}
        self.ref: dict[str, dict[str, float]] = {}     # robust stats of D and U per modality (validation)
        self.val_z: dict[str, dict[str, np.ndarray]] = {}   # scheme -> modality -> (n_val, 3) columns Q, zD, zU
        self.q_pool: dict[str, np.ndarray] = {}        # Q anomalies of the pooled non-test reference
        self._dist_cache: dict[tuple, dict] = {}
        self.calibrated = False

    # -- evidence ----------------------------------------------------------------------------
    @torch.no_grad()
    def modality_output(self, m: str, phys_m: np.ndarray, chunk: int = 256) -> ModalityOutput:
        x_norm = self.pp.normalize({m: phys_m}, self.norm)[m]
        enc = self.model.encoders[m]
        head = self.model.aux_heads[m]
        linear = head[1]
        p_feat = enc.feature_dropout.p
        p_head = head[0].p
        hs, auxs, us = [], [], []
        gen = torch.Generator().manual_seed(self.mc_seed)
        for i in range(0, x_norm.shape[0], chunk):
            xb = torch.from_numpy(x_norm[i:i + chunk])
            pooled, stats = enc.pooled_features(xb)
            h = enc.embed(pooled, stats, dropout_mask=torch.ones_like(pooled))
            hs.append(h.numpy())
            auxs.append(_softmax(linear(h)))
            # MC dropout with explicit masks -> identical inputs give identical evidence
            probs = []
            for _ in range(self.mc_samples):
                m1 = (torch.rand(pooled.shape, generator=gen) >= p_feat).float() / (1 - p_feat)
                ht = enc.embed(pooled, stats, dropout_mask=m1)
                m2 = (torch.rand(ht.shape, generator=gen) >= p_head).float() / (1 - p_head)
                probs.append(_softmax(linear(ht * m2)))
            pm = np.mean(probs, axis=0)
            us.append(entropy(pm) / math.log(self.task.n_classes))
        if self.quality:
            f = raw_features(m, phys_m, self.task.model_hz[m])
            q, z = self.quality["validation"][m].anomaly_from_features(f)
            qp, zp = self.quality["pooled"][m].anomaly_from_features(f)
        else:
            q = qp = np.zeros(x_norm.shape[0])
            f = z = zp = np.zeros((x_norm.shape[0], 1))
        return ModalityOutput(np.concatenate(hs), np.concatenate(auxs), np.concatenate(us), q, f, z, qp, zp)

    @torch.no_grad()
    def neutral_embedding(self, m: str) -> np.ndarray:
        """mu_m: embedding of a neutral window whose every sample equals the training mean (normalised
        zero), i.e. what the model sees for an unavailable stream. Chosen over the mean clean
        embedding, cross-modal reconstruction and leave-one-out heads in a design study on validation
        subjects only (see docs/METHODOLOGY.md)."""
        c, length = self.task.input_shapes()[m]
        enc = self.model.encoders[m]
        pooled, stats = enc.pooled_features(torch.zeros(1, c, length))
        return enc.embed(pooled, stats, dropout_mask=torch.ones_like(pooled)).numpy()[0]

    def outputs(self, phys: dict[str, np.ndarray]) -> dict[str, ModalityOutput]:
        return {m: self.modality_output(m, phys[m]) for m in self.modalities}

    def signal_quality(self, outs: dict[str, ModalityOutput], calibration: str | None = None) -> dict[str, np.ndarray]:
        return {m: o.quality(calibration or self.calibration)[0] for m, o in outs.items()}

    def uncertainty(self, outs: dict[str, ModalityOutput]) -> dict[str, np.ndarray]:
        return {m: o.u for m, o in outs.items()}

    def disagreement(self, outs: dict[str, ModalityOutput]) -> dict[str, np.ndarray]:
        if len(self.modalities) == 1:
            return {m: np.zeros(outs[m].aux.shape[0]) for m in self.modalities}
        out = {}
        for m in self.modalities:
            others = np.mean([outs[j].aux for j in self.modalities if j != m], axis=0)
            out[m] = js_divergence(outs[m].aux, others)
        return out

    def _z(self, m: str, d: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.ref[m]
        zd = np.clip((d - r["d_med"]) / r["d_scale"], *Z_CLIP)
        zu = np.clip((u - r["u_med"]) / r["u_scale"], *Z_CLIP)
        return zd, zu

    # -- calibration (clean validation only) --------------------------------------------------
    def calibrate(self, val_phys: dict[str, np.ndarray],
                  ref_phys: dict[str, np.ndarray] | None = None) -> "ReliabilityEngine":
        """``val_phys``: clean windows of the held-out validation subjects. They fit every
        reference of the report's ``validation`` scheme: Q indicators, D and U statistics and the
        reliability distribution.
        ``ref_phys``: clean windows of all non-test subjects, used only by the ``pooled`` extension
        for the model-free Q reference (defaults to the validation windows)."""
        ref_phys = ref_phys if ref_phys is not None else val_phys
        hz = self.task.model_hz
        self.quality = {
            "validation": {m: QualityCalibration(m, hz[m]).fit(val_phys[m]) for m in self.modalities},
            "pooled": {m: QualityCalibration(m, hz[m]).fit(ref_phys[m]) for m in self.modalities},
        }
        self.q_pool = {m: self.quality["pooled"][m].anomaly(ref_phys[m])[0].astype(np.float32) for m in self.modalities}
        outs = self.outputs(val_phys)
        d = self.disagreement(outs)
        self.val_z = {c: {} for c in CALIBRATIONS}
        for m in self.modalities:
            self.mu[m] = self.neutral_embedding(m)
            dm, um = d[m], outs[m].u
            self.ref[m] = {
                "d_med": float(np.median(dm)), "d_scale": float(max(np.subtract(*np.percentile(dm, [75, 25])) / 1.349, 1e-3)),
                "u_med": float(np.median(um)), "u_scale": float(max(np.subtract(*np.percentile(um, [75, 25])) / 1.349, 1e-3)),
            }
            zd, zu = self._z(m, dm, um)
            for c in CALIBRATIONS:
                self.val_z[c][m] = np.stack([outs[m].quality(c)[0], zd, zu], axis=1).astype(np.float32)
        self._dist_cache.clear()
        self.calibrated = True
        return self

    def _distribution(self, components: tuple[str, ...], calibration: str | None = None) -> dict:
        calibration = calibration or self.calibration
        if calibration not in CALIBRATIONS:
            raise ValueError(f"unknown calibration {calibration}; expected one of {CALIBRATIONS}")
        comps = tuple(sorted(components))
        key = (calibration, comps)
        if key not in self._dist_cache:
            cols = [COMPONENTS.index(c) for c in comps]
            q = 1.0 - self.alpha / len(self.modalities)
            dist = {}
            for m in self.modalities:
                if calibration == "pooled" and comps == ("Q",):
                    a = np.sort(self.q_pool[m]).astype(np.float32)   # subject-diverse reference (all non-test subjects)
                else:
                    a = np.sort(self.val_z[calibration][m][:, cols].sum(axis=1)).astype(np.float32)   # validation subjects
                n = a.size
                k = min(n - 1, int(math.ceil((n + 1) * q)) - 1)   # conformal (1 - alpha/M) quantile
                tau = float(a[k])
                med = float(np.median(a))
                dist[m] = {"sorted": a, "tau": tau, "median": med, "span": max(tau - med, 1.0) * self.gate_span}
            self._dist_cache[key] = dist
        return self._dist_cache[key]

    # -- scoring + reliability-aware fusion ----------------------------------------------------
    def score(self, outs: dict[str, ModalityOutput], components: tuple[str, ...] = DEFAULT_COMPONENTS,
              calibration: str | None = None):
        if not self.calibrated:
            raise RuntimeError("engine is not calibrated")
        calibration = calibration or self.calibration
        dist = self._distribution(components, calibration)
        d = self.disagreement(outs)
        evidence, anomaly, rel, weights, margin = {}, {}, {}, {}, {}
        for m in self.modalities:
            zd, zu = self._z(m, d[m], outs[m].u)
            qm = outs[m].quality(calibration)[0]
            parts = {"Q": qm, "D": zd, "U": zu}
            a = np.sum([parts[c] for c in components], axis=0) if components else np.zeros_like(zd)
            ref = dist[m]
            n = ref["sorted"].size
            # compare in the reference's float32 so exact ties count as ties (keeps R conservative)
            n_ge = n - np.searchsorted(ref["sorted"], a.astype(ref["sorted"].dtype), side="left")
            rel[m] = (1.0 + n_ge) / (n + 1.0)
            margin[m] = (a - ref["tau"]) / ref["span"]
            weights[m] = np.clip(1.0 - margin[m], 0.0, 1.0)
            anomaly[m] = a
            evidence[m] = {"Q": qm, "D": d[m], "U": outs[m].u, "zD": zd, "zU": zu}
        min_r = np.min(np.stack(list(rel.values()), axis=1), axis=1)
        flag = min_r <= self.alpha / len(self.modalities)
        return evidence, anomaly, rel, weights, flag, margin

    @torch.no_grad()
    def decide(self, outs: dict[str, ModalityOutput], components: tuple[str, ...] = DEFAULT_COMPONENTS,
               strategy: str = "embedding_gate", calibration: str | None = None) -> Decision:
        calibration = calibration or self.calibration
        evidence, anomaly, rel, weights, flag, margin = self.score(outs, components, calibration)
        h = {m: torch.from_numpy(outs[m].h) for m in self.modalities}
        fixed = _softmax(self.model.classify(self.model.fuse(h)))
        if strategy == "poe":
            logp = sum(weights[m][:, None] * np.log(np.clip(outs[m].aux, 1e-6, 1)) for m in self.modalities)
            logp = logp - logp.max(axis=1, keepdims=True)
            defended = np.exp(logp) / np.exp(logp).sum(axis=1, keepdims=True)
        else:
            w = {m: torch.from_numpy(weights[m].astype(np.float32)) for m in self.modalities}
            mu = {m: torch.from_numpy(self.mu[m].astype(np.float32)) for m in self.modalities}
            defended = _softmax(self.model.classify(self.model.fuse(h, weights=w, mu=mu)))
        return Decision(fixed_probs=fixed, defended_probs=defended,
                        aux_probs={m: outs[m].aux for m in self.modalities}, evidence=evidence,
                        anomaly=anomaly, reliability=rel, weights=weights, flag=flag,
                        components=tuple(components), margin=margin, calibration=calibration)

    def run(self, phys: dict[str, np.ndarray], **kw) -> Decision:
        return self.decide(self.outputs(phys), **kw)

    # -- persistence ---------------------------------------------------------------------------
    def state(self) -> dict:
        return {
            "alpha": self.alpha, "gate_span": self.gate_span, "mc_samples": self.mc_samples,
            "quality": {c: {m: q.to_dict() for m, q in qs.items()} for c, qs in self.quality.items()},
            "mu": {m: v.tolist() for m, v in self.mu.items()},
            "ref": self.ref,
            "val_z": {c: {m: v.tolist() for m, v in vs.items()} for c, vs in self.val_z.items()},
            "q_pool": {m: v.tolist() for m, v in self.q_pool.items()},
        }

    def load_state(self, s: dict) -> "ReliabilityEngine":
        self.alpha = s["alpha"]
        self.gate_span = s["gate_span"]
        self.mc_samples = s["mc_samples"]
        self.quality = {c: {m: QualityCalibration.from_dict(q) for m, q in qs.items()} for c, qs in s["quality"].items()}
        self.mu = {m: np.asarray(v, dtype=np.float32) for m, v in s["mu"].items()}
        self.ref = s["ref"]
        self.val_z = {c: {m: np.asarray(v, dtype=np.float32) for m, v in vs.items()} for c, vs in s["val_z"].items()}
        self.q_pool = {m: np.asarray(v, dtype=np.float32) for m, v in s["q_pool"].items()}
        self._dist_cache.clear()
        self.calibrated = True
        return self

    def _hist_source(self, m: str, c: str, i: int, calibration: str) -> np.ndarray:
        """Clean reference values of one evidence channel. Q under the pooled scheme is the pooled
        non-test reference (clipped at its 99.5th percentile for display); everything else is the
        validation subjects' z-scores."""
        if c == "Q" and calibration == "pooled" and self.q_pool:
            q = self.q_pool[m]
            return np.clip(q, None, np.percentile(q, 99.5))
        return self.val_z[calibration][m][:, i]

    def calibration_summary(self, components: tuple[str, ...] = DEFAULT_COMPONENTS,
                            calibration: str | None = None) -> dict:
        calibration = calibration or self.calibration
        dist = self._distribution(components, calibration)
        out = {}
        for m in self.modalities:
            v = self.val_z[calibration][m]
            hists = {c: np.histogram(self._hist_source(m, c, i, calibration), bins=30) for i, c in enumerate(COMPONENTS)}
            out[m] = {
                "tau": dist[m]["tau"], "median": dist[m]["median"], "span": dist[m]["span"],
                "n_val": int(v.shape[0]), "n_ref": int(dist[m]["sorted"].size), "calibration": calibration,
                "anomaly_quantiles": np.quantile(dist[m]["sorted"], [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).tolist(),
                "quality": self.quality[calibration][m].to_dict(), "ref": self.ref[m],
                "hist": {c: h[0].tolist() for c, h in hists.items()},
                "hist_edges": {c: h[1].tolist() for c, h in hists.items()},
            }
        return out
