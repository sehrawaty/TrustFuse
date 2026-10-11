# TrustFuse methodology

This document describes what the code in `backend/trustfuse/` implements. It follows the
project report ("Security Evaluation of Intentional Modality Suppression in Wearable Health AI",
Review 2). It also records every decision taken while building the system and how that decision
was made.

## 1. Research question

Robustness to *naturally* missing data does not establish robustness against *intentional*
suppression. The study asks:

1. Does deliberately suppressing one sensor stream measurably degrade a multimodal wearable
   model?
2. Do the modalities differ in how vulnerable they are?
3. Can reliability-aware fusion detect the resulting evidence imbalance and reduce the damage
   without materially hurting clean performance?

The contribution is a security-oriented empirical characterisation. The building blocks
(missing-modality handling, reliability weighting, multimodal fusion) are not claimed to be
individually novel.

## 2. Data

| | WESAD (primary) | PPG-DaLiA (validation) |
|---|---|---|
| Subjects | 15 (S2–S11, S13–S17) | 15 (S1–S15) |
| Streams used | wrist BVP 64 Hz, ACC 32 Hz, EDA 4 Hz, TEMP 4 Hz | wrist BVP 64 Hz, ACC 32 Hz |
| Task(s) | stress vs. non-stress (headline); baseline / stress / amusement | 8 daily activities |
| Window / stride | 30 s / 5 s | 8 s / 4 s |
| Model input rates | BVP 32, ACC 16, EDA 4, TEMP 4 Hz | BVP 32, ACC 32 Hz |

* The stored files keep the native wrist streams in physical units (`data/processed`, see
  `data/DATA_LICENSE.md`). ACC is stored losslessly as int8 counts of 1/64 g.
* Windows are **label-pure**: every 0.25 s label inside a window has the same class. They are
  placed on a global stride grid.
* Polyphase resampling (`scipy.signal.resample_poly`) converts native rates to model rates.

## 3. Evaluation protocol and leakage control

* **Leave-one-subject-out (LOSO).** Each subject is the test set of exactly one fold.
* **Validation subjects.** In each fold, the next two subjects in canonical order are held out
  for validation, so validation is subject-disjoint and deterministic. A unit test asserts that
  train, validation and test never share a subject.
* **Normalisation** uses z-scores computed from the fold's training windows only.
* **Early stopping** uses validation macro-F1.
* **Reliability calibration** uses only clean data from non-test subjects (§5.4). It is frozen
  before any attack is evaluated.
* **Matched comparison.** The fixed and reliability-aware pipelines use the same frozen
  checkpoint and receive byte-identical attacked windows.
* **Pre-registration.** Everything is fixed in `configs/*.yaml` before test evaluation: the
  attack grid, α, the non-inferiority margin and the false-rejection limit. Each result set
  stores its configuration and a configuration hash.

## 4. Models

**FusionModel** (`models/fusion.py`):
* Each modality has a compact 1-D CNN encoder E_m: stride-2 conv blocks, then mean+max pooling
  plus window statistics, projected to a 32-d embedding h_m.
* **Standard (fixed) fusion** concatenates the embeddings and applies a two-layer MLP.
* Every modality also has an **auxiliary unimodal head** g_m(h_m). These heads are trained
  jointly (deep supervision, weight 0.5); they supply per-modality evidence without changing the
  fusion path.
* Size: about 134 k parameters.

Baselines:
* **Unimodal models**: the same encoder plus an MLP, one per modality.
* **ModDrop (extension)**: the FusionModel trained with random modality dropout (p = 0.5). It
  represents "robust to generic missingness" training.

Training settings:
* AdamW (lr 2e-3, weight decay 1e-4), cosine schedule, class-weighted cross-entropy.
* Up to 40 epochs, patience 7, batch size 64.
* Fixed seeds: 3 seeds for the headline fusion model, 1 for the other models.
* Checkpoints are stored in fp16.

## 5. Selective Modality Suppression Attack (SMSA)

`attacks/simulator.py`. For a target modality m, the transform A_m changes only x_m. Untargeted
streams are passed through unchanged; unit tests assert byte identity. Attacks act on
synchronized windows in physical units, at the attack surface between preprocessing and the
encoders (report Fig. 2).

| Primitive | Definition | Grid |
|---|---|---|
| Masking | A contiguous segment covering a fraction *s* of the window is replaced by **zero** (the sensor reports zeros), **missing** (NaN; the model imputes the training mean) or **hold** (frozen at the last valid value). | s ∈ {10, 20, 40, 60, 80, 100} %; all three fills |
| Attenuation | x_m ← (1 − s)·x_m | s ∈ {10, 20, 40, 60, 80} % |
| Burst | The continuous stream is suppressed over B consecutive windows, then restored. Windows at the edges are partially affected. | B ∈ {1, 2, 5, 10}; 12 random placements per subject |
| Noise (extension) | Additive Gaussian noise at a set SNR | {20, 10, 5, 0} dB |

Attackers. Each suppresses one modality per window, fully masked:

* **Random**: a uniformly random modality. This is the benign "missing modality" case.
* **Targeted**: label-free and query-based. It picks the modality whose removal changes the
  fixed-fusion output most (total variation).
* **Oracle**: label-aware worst case. It picks the modality that minimises the probability of
  the true class.
* **Adaptive (extension, defense-aware)**: searches 40 candidates per modality (3 fills ×
  10 severities, plus 10 attenuations). It takes the cheapest candidate that flips the
  *defended* decision without raising the flag. If no candidate does, it takes the most
  damaging unflagged candidate.

Every attacked window is written to `manifest.npz` with its decisions, reliability and flag.

## 6. Reliability-aware fusion

### 6.1 Signal quality Q (model-free)

Indicators per modality, fixed from sensor-quality practice:
* missing share;
* exact-zero share (contact loss);
* share of non-changing samples;
* longest constant run (frozen sensor);
* physical implausibility: skin temperature outside 20–42 °C, EDA outside 0–100 µS, ACC
  magnitude outside 0.3–3.5 g;
* window level (TEMP, ACC magnitude);
* BVP only: log amplitude and cardiac-band (0.7–3.5 Hz) spectral power share.

Each indicator becomes a *directional* robust z-score: (value − median) / (IQR / 1.349), keeping
only movement in the suspicious direction. Q_m is the largest of these z-scores.

The Q reference pools clean windows of **all non-test subjects** (training and validation; up to
300 per subject). Physiological levels vary strongly between people. A reference built from the
two validation subjects alone produced false alarms on clean windows of new subjects (§9).

### 6.2 Disagreement D

D_m = JS( g_m(h_m) ‖ mean_{j≠m} g_j(h_j) ), the Jensen–Shannon divergence in bits between the
unimodal head of m and the consensus of the other heads.

### 6.3 Uncertainty U

U_m is the entropy of the mean over T = 8 MC-dropout passes of head m, normalised by log K.
Dropout masks come from a seeded generator, so identical inputs always give identical evidence.

### 6.4 Conformal reliability, flag and gate

* **Evidence set E.** The anomaly of modality m is a_m = Σ_{c∈E} z_c. The **default operating
  point is E = {Q}** (§6.6). Q is ≥ 0 by construction. z(D) and z(U) are robust z-scores against
  the clean **validation** windows, clipped to [−3, 50]; the validation subjects were not used
  for fitting.
* **Reliability.** R_m = (1 + #{reference windows with a ≥ a_m}) / (n_ref + 1) is a conformal
  p-value. On clean, exchangeable data P(R ≤ t) ≤ t (unit-tested). Comparisons happen in the
  reference's float32 precision so ties stay ties.
* **Reference set.** For E = {Q} it is every non-test subject (up to 300 windows each), since Q is
  model-free. For evidence sets that include the model-based D or U, it is the held-out
  validation subjects.
* **Low-confidence flag**: min_m R_m ≤ α / M, with α = 0.05 (Bonferroni over the M modalities).
* **Gate**: w_m = clip(1 − (a_m − τ_m) / (τ_m − median a_m), 0, 1), where τ_m is the conformal
  (1 − α/M) quantile of the reference.
* **Defended representation**: h̃_m = w_m h_m + (1 − w_m) μ_m. It passes through the *same
  frozen* classifier.
* **μ_m** is the embedding of a neutral window whose every sample equals the training mean,
  i.e. exactly what the model sees for a missing stream.
* **Detection score** (for ROC analysis): the largest calibrated margin (a_m − τ_m)/(τ_m − median).
  Unlike −log R it does not saturate at the conformal floor 1/(n+1).

### 6.5 How μ_m was chosen (validation-only design study)

Four substitutes for a gated-out modality were compared with **oracle gating** (w = 0 on the
attacked modality), which isolates substitute quality from detection. The comparison ran on the
*validation subjects* of four folds; no test subject was used. Full zero-masking, mean macro-F1
over modalities:

| Substitute | BVP | ACC | EDA | TEMP | Mean |
|---|---|---|---|---|---|
| No defense (fixed fusion) | 0.766 | 0.870 | 0.689 | 0.723 | 0.762 |
| Mean clean embedding | 0.646 | 0.880 | 0.873 | 0.881 | 0.820 |
| Cross-modal ridge reconstruction | 0.710 | 0.890 | 0.856 | 0.889 | 0.836 |
| Leave-one-out heads on frozen embeddings | 0.680 | 0.865 | 0.832 | 0.822 | 0.800 |
| **Neutral-input embedding (adopted)** | **0.766** | 0.865 | 0.871 | **0.895** | **0.849** |

### 6.6 Why the default evidence is subject-calibrated Q (validation-only study)

**What prompted the study.** A preliminary run of the supplementary 3-class experiment, with
E = {Q, D, U} calibrated on the two validation subjects, flagged about 22 % of *clean* windows of
held-out people. The flag is meant to stay near α = 5 %. The run was diagnosed and then discarded.

**Protocol.** To choose a fix without using test subjects, each fold's two validation subjects
were swapped: calibrate on one, measure on the other. This reproduced the problem (18 % clean
false flags) and allowed the calibration schemes to be compared. Results over 6 WESAD folds × 2
directions; the clean column is the false-flag rate, the others are attack detection rates:

| Scheme | Clean false flags | Zero-mask 100 % | Zero-mask 40 % | Hold 100 % | Attenuate 80 % |
|---|---|---|---|---|---|
| Q+D+U, reference = one held-out subject | 18.4 % (max 71 %) | 100 % | 95.5 % | 100 % | 65.2 % |
| **Q, reference pooled over the other non-test subjects** | **6.1 %** (max 16 %) | 100 % | 100 % | 100 % | 59.0 % |
| min(Q pooled, D+U held-out) | 24.8 % | 100 % | 84.9 % | 100 % | 70.9 % |
| Q pooled ∨ D+U held-out (α split 3:1) | 22.8 % | 100 % | 100 % | 100 % | 69.6 % |

**Finding.** Model-based evidence (disagreement, uncertainty) calibrated on a few people does not
transfer to new people. The weak unimodal heads, especially TEMP and EDA, disagree with the others
in a person-specific way. Model-free quality evidence calibrated against many people stays close
to its nominal false-alarm rate and loses nothing on masking attacks. The canonical evaluation
therefore uses E = {Q}. D and U are still computed for every window and shown in the UI, and the
ablations evaluate every evidence subset (Q+D+U, Q+D, Q+U, D+U, D, U).

## 7. Metrics

All metrics are computed per held-out subject:
* accuracy, macro-F1, balanced accuracy;
* ΔF1 versus clean;
* **attack success rate**: the share of windows correct on clean data that become wrong under
  attack;
* flip rate;
* **confidence distortion**: mean confidence, the "confidently wrong" share (wrong with
  confidence ≥ 0.8), and ECE;
* **false-rejection rate**: the flag rate on clean windows;
* detection AUROC: the suspicion −log min R on attacked versus clean windows, both overall and
  per evidence channel (the detectability matrix);
* defense gain = defended F1 − fixed F1;
* clean cost = fixed F1 − defended F1, on clean data;
* the **Suppression Vulnerability Index**: the normalised area under the macro-F1-drop versus
  severity curve.

## 8. Statistics and hypotheses

The subject is the unit of analysis. Seeds are averaged within a subject first. 95 % CIs use a
subject-cluster bootstrap with 2,000 resamples. Effect sizes are matched-pairs rank-biserial r
and Cohen's d_z.

| | Hypothesis | Test | Supported if |
|---|---|---|---|
| H1 | Full suppression of a modality lowers fixed-fusion macro-F1 | one-sided paired Wilcoxon, Holm over modalities | at least one modality significant |
| H2 | Modalities differ in vulnerability | Friedman on per-subject drops (Kendall's W); Holm post-hoc | p < α |
| H3 | Defended > fixed under the same attack, with clean cost ≤ 0.02 macro-F1 | one-sided paired Wilcoxon, Holm; bootstrap non-inferiority (upper CI ≤ margin) | non-inferior and at least one H1-vulnerable modality mitigated |
| H4 | Suppression is detectable (AUROC > 0.5) at a clean FRR ≤ 0.10 | one-sided one-sample Wilcoxon of per-subject AUROC vs 0.5, Holm; bootstrap FRR | every modality detectable and mean FRR within the limit |

Secondary analyses:
* targeted versus random suppression (one-sided Wilcoxon);
* the attacker ladder;
* ModDrop versus defended fusion;
* Q/D/U ablations and a decision-level product-of-experts variant.

## 9. Decisions made during development (and why)

1. **Subject-diverse calibration** (§6.6).
2. **Q reference pooling.** The first calibration used only the two validation subjects. On the
   clean held-out subject of the first fold this flagged 88 % of windows: S2's skin temperature
   (≈35 °C) lay outside the band of the two reference subjects. Model-free indicators now use a
   reference pooled over all non-test subjects. Data-driven amplitude bounds were replaced by
   physical plausibility bounds. No test subject is involved in either change.
3. **Neutral substitute** μ_m: chosen in the validation-only study in §6.5.
4. **Detection score.** −log R saturates for strongly suppressed windows, which made per-modality
   AUROCs identical in the preliminary run. ROC analysis therefore uses the calibrated anomaly
   margin; the flag itself is unchanged.
5. **Memory budget.** Several measures keep the API container under its 900 MB limit, with a
   measured peak of about 650 MB during a live training job:
   * calibration references are subsampled per subject;
   * the decoded-stream cache is bounded;
   * glibc is configured to return large buffers to the OS immediately;
   * calibrations of the shipped models are precomputed.

## 10. Reproducing

```
cd backend
pip install -r requirements-dev.txt
python -m trustfuse verify                                  # checksums of data/processed
python -m trustfuse train wesad_binary wesad_3class dalia_activity --workers 3
python -m trustfuse evaluate wesad_binary --workers 4       # -> artifacts/results/wesad_binary
python -m pytest -q
```

The canonical result sets in `artifacts/results/` were produced with exactly these commands and
the configurations in `configs/`.
