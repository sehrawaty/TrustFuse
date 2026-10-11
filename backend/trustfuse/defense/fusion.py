"""Reliability-aware fusion (the defense).

The defended and undefended pipelines share the same clean-trained checkpoint and receive the
same (possibly attacked) windows; the defense differs only in how modality embeddings are fused:

    fixed fusion       logits = f([h_1, ..., h_M])
    reliability-aware  logits = f([w_1 h_1 + (1-w_1) mu_1, ..., w_M h_M + (1-w_M) mu_M])

with w_m derived from the calibrated reliability R_m (see ``reliability.engine``). A
decision-level alternative (reliability-weighted product of the unimodal experts) is available
for ablation as ``strategy="poe"``.
"""

from __future__ import annotations

import numpy as np

from ..reliability.engine import DEFAULT_COMPONENTS, Decision, ReliabilityEngine


class ReliabilityAwareFusion:
    def __init__(self, engine: ReliabilityEngine, components: tuple[str, ...] = DEFAULT_COMPONENTS,
                 strategy: str = "embedding_gate"):
        self.engine = engine
        self.components = tuple(components)
        self.strategy = strategy

    def __call__(self, phys: dict[str, np.ndarray]) -> Decision:
        return self.engine.decide(self.engine.outputs(phys), components=self.components, strategy=self.strategy)
