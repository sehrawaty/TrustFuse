"""Pick the guided-demo window from the canonical WESAD result set: a window of the requested
subject where full zero-masking of the target modality flips the fixed-fusion decision, the
reliability-aware pipeline keeps the correct one, the flag fires, and the clean decision was
correct and not flagged. Prints the window index used in web/src/lib/demo.ts.

usage: python find_demo_window.py [subject] [modality] [current_index]
With ``current_index`` the script also reports whether that window still qualifies."""

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
ok = ((clean[0] == y) & (clean[1] == y) & (clean[6] == 0)
      & (attacked[0] != y) & (attacked[1] == y) & (attacked[6] == 1))
idx = np.flatnonzero(ok)
windows = load_stream("wesad_binary", subject).windows.start_s.tolist()
print(f"{subject}: {len(y)} windows, clean windows flagged {float(np.mean(clean[6])):.1%}")
if len(sys.argv) > 3:
    cur = int(sys.argv[3])
    pos = int(np.flatnonzero(start == windows[cur])[0])
    print(f"current index {cur}: qualifies={bool(ok[pos])} (clean fixed/def {int(clean[0][pos])}/{int(clean[1][pos])}, "
          f"clean flag {int(clean[6][pos])}, attacked fixed/def {int(attacked[0][pos])}/{int(attacked[1][pos])}, "
          f"attacked flag {int(attacked[6][pos])}, y={int(y[pos])})")
if not idx.size:
    sys.exit(f"no qualifying window for {subject}/{modality}")
stress = idx[y[idx] == 1]
pick = int((stress if stress.size else idx)[len(stress if stress.size else idx) // 2])
print(f"{subject} {modality}: {idx.size} qualifying windows; pick index {windows.index(int(start[pick]))} (t={start[pick]} s, y={y[pick]})")
