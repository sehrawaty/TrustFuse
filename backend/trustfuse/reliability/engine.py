"""ReliabilityEngine: per-modality evidence -> reliability scores -> fusion weights.

Evidence for modality m of a window:
    Q_m  signal-quality anomaly       (model-free, see ``quality.py``)
    D_m  cross-modal disagreement     JS( g_m(h_m) || mean_{j != m} g_j(h_j) )   [bits]
    U_m  predictive uncertainty       entropy of the MC-dropout mean of g_m, normalised by log K

Calibration uses clean data of non-test subjects only and is frozen before any attack is
evaluated: the model-free Q reference pools training + validation subjects, while the
model-based D and U statistics and the reliability distribution use the held-out validation
subjects (the model was not fitted on them).
D and U are turned into robust z-scores against their clean-validation distributions and summed
with the Q anomaly into an anomaly score a_m. The reliability score

    R_m = (1 + #{ clean-validation windows with a >= a_m }) / (n + 1)

is a conformal p-value: for clean, exchangeable data P(R_m <= t) <= t. The low-confidence flag
fires when min_m R_m <= alpha / M (Bonferroni over modalities), so its clean false-rejection
rate is bounded by alpha. The fusion weight

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
from .quality import QualityCalibration

COMPONENTS = ("Q", "D", "U")
Z_CLIP = (-3.0, 50.0)


@dataclass
class ModalityOutput:
    h: np.ndarray          # (B, E) deterministic embedding
    aux: np.ndarray        # (B, K) deterministic unimodal-head probabilities
    u: np.ndarray          # (B,)   normalised predictive entropy (MC dropout)
    q: np.ndarray          # (B,)   signal-quality anomaly (>= 0)
    q_feat: np.ndarray     # (B, F) raw quality indicators
    q_z: np.ndarray        # (B, F) directional z-scores of the indicators

    def take(self, idx) -> "ModalityOutput":
        return ModalityOutput(self.h[idx], self.aux[idx], self.u[idx], self.q[idx], self.q_feat[idx], self.q_z[idx])


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
    components: tuple[str, ...] = field(default=COMPONENTS)

    @property
    def fixed_pred(self) -> np.ndarray:
        return self.fixed_probs.argmax(1)

    @property
    def defended_pred(self) -> np.ndarray:
        return self.defended_probs.argmax(1)

    def min_reliability(self) -> np.ndarray:
        return np.min(np.stack(list(self.reliability.values()), axis=1), axis=1)

    def suspicion(self) -> np.ndarray:
        """Detection score: larger = more suspicious (used for ROC analysis)."""
        return -np.log(self.min_reliability())


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
        self.quality: dict[str, QualityCalibration] = {}
        self.mu: dict[str, np.ndarray] = {}
        self.ref: dict[str, dict[str, float]] = {}     # robust stats of D and U per modality
        self.val_z: dict[str, np.ndarray] = {}         # (n_val, 3) columns Q, zD, zU per modality
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
            q, f, z = self.quality[m].anomaly(phys_m)
        else:
            q = np.zeros(x_norm.shape[0])
            f = z = np.zeros((x_norm.shape[0], 1))
        return ModalityOutput(np.concatenate(hs), np.concatenate(auxs), np.concatenate(us), q, f, z)

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

    def signal_quality(self, outs: dict[str, ModalityOutput]) -> dict[str, np.ndarray]:
        return {m: o.q for m, o in outs.items()}

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
        """``val_phys``: clean windows of the held-out validation subjects (D, U, R calibration).
        ``ref_phys``: clean windows of all non-test subjects for the model-free Q reference
        (defaults to the validation windows)."""
        ref_phys = ref_phys if ref_phys is not None else val_phys
        self.quality = {m: QualityCalibration(m, self.task.model_hz[m]).fit(ref_phys[m]) for m in self.modalities}
        outs = self.outputs(val_phys)
        d = self.disagreement(outs)
        for m in self.modalities:
            self.mu[m] = self.neutral_embedding(m)
            dm, um = d[m], outs[m].u
            self.ref[m] = {
                "d_med": float(np.median(dm)), "d_scale": float(max(np.subtract(*np.percentile(dm, [75, 25])) / 1.349, 1e-3)),
                "u_med": float(np.median(um)), "u_scale": float(max(np.subtract(*np.percentile(um, [75, 25])) / 1.349, 1e-3)),
            }
            zd, zu = self._z(m, dm, um)
            self.val_z[m] = np.stack([outs[m].q, zd, zu], axis=1).astype(np.float32)
        self._dist_cache.clear()
        self.calibrated = True
        return self

    def _distribution(self, components: tuple[str, ...]) -> dict:
        key = tuple(sorted(components))
        if key not in self._dist_cache:
            cols = [COMPONENTS.index(c) for c in key]
            q = 1.0 - self.alpha / len(self.modalities)
            dist = {}
            for m in self.modalities:
                a = np.sort(self.val_z[m][:, cols].sum(axis=1))
                n = a.size
                k = min(n - 1, int(math.ceil((n + 1) * q)) - 1)   # conformal (1 - alpha/M) quantile
                tau = float(a[k])
                med = float(np.median(a))
                dist[m] = {"sorted": a, "tau": tau, "median": med, "span": max(tau - med, 1.0) * self.gate_span}
            self._dist_cache[key] = dist
        return self._dist_cache[key]

    # -- scoring + reliability-aware fusion ----------------------------------------------------
    def score(self, outs: dict[str, ModalityOutput], components: tuple[str, ...] = COMPONENTS):
        if not self.calibrated:
            raise RuntimeError("engine is not calibrated")
        dist = self._distribution(components)
        d = self.disagreement(outs)
        evidence, anomaly, rel, weights = {}, {}, {}, {}
        for m in self.modalities:
            zd, zu = self._z(m, d[m], outs[m].u)
            parts = {"Q": outs[m].q, "D": zd, "U": zu}
            a = np.sum([parts[c] for c in components], axis=0) if components else np.zeros_like(zd)
            ref = dist[m]
            n = ref["sorted"].size
            n_ge = n - np.searchsorted(ref["sorted"], a, side="left")
            rel[m] = (1.0 + n_ge) / (n + 1.0)
            weights[m] = np.clip(1.0 - (a - ref["tau"]) / ref["span"], 0.0, 1.0)
            anomaly[m] = a
            evidence[m] = {"Q": outs[m].q, "D": d[m], "U": outs[m].u, "zD": zd, "zU": zu}
        min_r = np.min(np.stack(list(rel.values()), axis=1), axis=1)
        flag = min_r <= self.alpha / len(self.modalities)
        return evidence, anomaly, rel, weights, flag

    @torch.no_grad()
    def decide(self, outs: dict[str, ModalityOutput], components: tuple[str, ...] = COMPONENTS,
               strategy: str = "embedding_gate") -> Decision:
        evidence, anomaly, rel, weights, flag = self.score(outs, components)
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
                        components=tuple(components))

    def run(self, phys: dict[str, np.ndarray], **kw) -> Decision:
        return self.decide(self.outputs(phys), **kw)

    # -- persistence ---------------------------------------------------------------------------
    def state(self) -> dict:
        return {
            "alpha": self.alpha, "gate_span": self.gate_span, "mc_samples": self.mc_samples,
            "quality": {m: q.to_dict() for m, q in self.quality.items()},
            "mu": {m: v.tolist() for m, v in self.mu.items()},
            "ref": self.ref,
            "val_z": {m: v.tolist() for m, v in self.val_z.items()},
        }

    def load_state(self, s: dict) -> "ReliabilityEngine":
        self.alpha = s["alpha"]
        self.gate_span = s["gate_span"]
        self.mc_samples = s["mc_samples"]
        self.quality = {m: QualityCalibration.from_dict(q) for m, q in s["quality"].items()}
        self.mu = {m: np.asarray(v, dtype=np.float32) for m, v in s["mu"].items()}
        self.ref = s["ref"]
        self.val_z = {m: np.asarray(v, dtype=np.float32) for m, v in s["val_z"].items()}
        self._dist_cache.clear()
        self.calibrated = True
        return self

    def calibration_summary(self, components: tuple[str, ...] = COMPONENTS) -> dict:
        dist = self._distribution(components)
        out = {}
        for m in self.modalities:
            v = self.val_z[m]
            out[m] = {
                "tau": dist[m]["tau"], "median": dist[m]["median"], "span": dist[m]["span"],
                "n_val": int(v.shape[0]),
                "anomaly_quantiles": np.quantile(dist[m]["sorted"], [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).tolist(),
                "quality": self.quality[m].to_dict(), "ref": self.ref[m],
                "hist": {c: np.histogram(v[:, i], bins=30)[0].tolist() for i, c in enumerate(COMPONENTS)},
                "hist_edges": {c: np.histogram(v[:, i], bins=30)[1].tolist() for i, c in enumerate(COMPONENTS)},
            }
        return out
