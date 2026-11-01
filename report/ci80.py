"""Cluster-aware uncertainty for the 80-pair mechanism in section 4.

The 80 (model, problem) pairs are 10 problems x 8 models, so the resampling
unit is the problem, not the pair.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from aimo_reference import pairs  # noqa: E402

full = pd.read_parquet(ROOT / "data-hf" / "aimo-interp-challenge-sample-full.parquet")
fr = pairs(full)
print("pairs:", len(fr), "problems:", fr.original_problem.nunique(),
      "models:", fr.model_id.nunique(), "robust:", int(fr.is_robust.sum()))

y = fr.is_robust.to_numpy().astype(int)
base = fr.base.to_numpy()
extreme = np.maximum(base, 1 - base)
probs = fr.original_problem.to_numpy()
uniq = np.unique(probs)

def stats(idx):
    yy, bb, ee = y[idx], base[idx], extreme[idx]
    if yy.min() == yy.max():
        return None
    return (
        roc_auc_score(yy, -bb),
        roc_auc_score(yy, ee),
        float(((bb == 0).astype(int) == yy).mean()),
        roc_auc_score(yy, (bb == 0).astype(float)),
    )

point = stats(np.arange(len(y)))
names = ["AUC(-base_accuracy)", "AUC(extremeness)", "acc(base==0 rule)", "AUC(base==0 rule)"]

rng = np.random.default_rng(20260919)
draws = []
for _ in range(5000):
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    idx = np.concatenate([np.flatnonzero(probs == p) for p in pick])
    s = stats(idx)
    if s is not None:
        draws.append(s)
draws = np.array(draws)
print("\ncluster bootstrap by problem, 5000 resamples, n_clusters =", len(uniq))
for i, nm in enumerate(names):
    lo, hi = np.percentile(draws[:, i], [2.5, 97.5])
    print(f"{nm:<22} {point[i]:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

print("\nsmall-model slice (<=10B aliases present in this split):")
for mid, g in fr.groupby("model_id"):
    print(f"  {mid:<45} pairs={len(g):>2} robust={int(g.is_robust.sum())} "
          f"base==0={int((g.base == 0).sum())} mean_base={g.base.mean():.2f}")

print("\ncontingency, base_accuracy x label:")
bucket = np.where(base == 0, "0.0", np.where(base == 1, "1.0", "intermediate"))
print(pd.crosstab(pd.Series(bucket, name="base"), pd.Series(y, name="robust")))
