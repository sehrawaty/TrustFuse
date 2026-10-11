"""Pick the guided-demo window from the canonical WESAD result set: a window of the requested
subject where full zero-masking of the target modality flips the fixed-fusion decision, the
reliability-aware pipeline keeps the correct one, the flag fires, and the clean decision was
correct. Prints the window index used in web/src/lib/demo.ts."""

import sys

import numpy as np

from trustfuse import paths
from trustfuse.data.preprocess import load_stream

subject = sys.argv[1] if len(sys.argv) > 1 else "S2"
modality = sys.argv[2] if len(sys.argv) > 2 else "eda"
with np.load(paths.ARTIFACTS_DIR / "results" / "wesad_binary" / "manifest.npz") as z:
    conds = z["conditions"].tolist()
    sel = z["subject"] == subject
    attacked = z[f"c{conds.index(f'mask|modality={modality}|fill=zero|severity=1')}"][:, sel]
    clean = z[f"c{conds.index('clean')}"][:, sel]
    y = z["y"][sel]
    start = z["start_s"][sel]
ok = (clean[0] == y) & (clean[1] == y) & (attacked[0] != y) & (attacked[1] == y) & (attacked[6] == 1)
idx = np.flatnonzero(ok)
if not idx.size:
    sys.exit(f"no qualifying window for {subject}/{modality}")
stress = idx[y[idx] == 1]
pick = int((stress if stress.size else idx)[len(stress if stress.size else idx) // 2])
windows = load_stream("wesad_binary", subject).windows.start_s.tolist()
print(f"{subject} {modality}: {idx.size} qualifying windows; pick index {windows.index(int(start[pick]))} (t={start[pick]} s, y={y[pick]})")
