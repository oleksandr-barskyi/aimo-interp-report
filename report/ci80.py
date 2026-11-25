"""Cluster-aware uncertainty for the 80-pair mechanism in section 4.

The 80 (model, problem) pairs are 10 problems x 8 models, so the resampling
unit is the problem, not the pair.  Every number is given under two label
rules: the zero-decay rule of the original report and the official
train-main-v2 rule (max_drop <= 0.10 robust, >= 0.25 not robust, band dropped).
The last block bounds the same mechanism on train-main-v2 itself, the only
labelled data on the five test checkpoints.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from aimo_reference import pairs  # noqa: E402

SMALL = {"qwen3-8b:low"}
full = pd.read_parquet(ROOT / "data-hf" / "aimo-interp-challenge-sample-full.parquet")
names = ["AUC(-base_accuracy)", "AUC(extremeness)", "acc(base==0 rule)", "AUC(base==0 rule)"]


def run(label):
    fr = pairs(full, label)
    print(f"\n==================== label rule: {label} ====================")
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

    print("\nper model:")
    for mid, g in fr.groupby("model_id"):
        print(f"  {mid:<45} pairs={len(g):>2} robust={int(g.is_robust.sum())} "
              f"base==0={int((g.base == 0).sum())} robust&base==0={int((g.is_robust & (g.base == 0)).sum())} "
              f"mean_base={g.base.mean():.2f}")

    bucket = np.where(base == 0, "0.0", np.where(base == 1, "1.0", "intermediate"))
    for name, sel in (("all models", np.ones(len(fr), bool)),
                      ("large/frontier models (7)", ~fr.model_id.isin(SMALL).to_numpy()),
                      ("qwen3-8b only (<=10B)", fr.model_id.isin(SMALL).to_numpy())):
        print(f"\ncontingency, base_accuracy x label, {name}:")
        print(pd.crosstab(pd.Series(bucket[sel], name="base"), pd.Series(y[sel], name="robust")))
    return fr


run("zero")
run("official")

print("\n==================== train-main-v2 (the five test checkpoints) ====================")
t = pd.read_parquet(ROOT / "data-hf" / "train-main-v2.parquet")
lab = t[t.is_robust.notna()].copy()
lab["is_robust"] = lab.is_robust.astype(bool)
r = lab[lab.is_robust]
print(f"rows {len(t)}, labelled {len(lab)}, robust {len(r)}, unlabelled {t.is_robust.isna().sum()}, "
      f"problems {t.problem.nunique()}")
print("base_accuracy is not published; base == 0 forces max_drop <= 0, so the robust rows with")
print("max_drop <= 0 bound from above how many robust rows can sit at base accuracy 0:")
print(f"  robust rows with max_drop <= 0: {(r.max_drop <= 0).sum()}/{len(r)} "
      f"(max_drop < 0: {(r.max_drop < 0).sum()}, == 0: {(r.max_drop == 0).sum()})")
print(f"  robust rows with 0 < max_drop <= 0.10 (base > 0 for certain): {(r.max_drop > 0).sum()}/{len(r)}")
print("per configuration:")
for (mid, eff), g in lab.groupby(["model_id", "reasoning_effort"]):
    rr = g[g.is_robust]
    print(f"  {mid + ' ' + eff:<50} robust={len(rr):>2} of {len(g):>2}  max_drop<=0 among robust={int((rr.max_drop <= 0).sum())}")

print("\nproblem memorisation: label of a row predicted by the majority label of the other")
print("configurations on the same problem text (ties and singleton problems skipped):")
hit = n = 0
for i, row in lab.iterrows():
    other = lab[(lab.problem == row.problem) & (lab.index != i)]
    if len(other) == 0 or other.is_robust.mean() == 0.5:
        continue
    n += 1
    hit += int((other.is_robust.mean() > 0.5) == row.is_robust)
print(f"  {hit} of {n}")
