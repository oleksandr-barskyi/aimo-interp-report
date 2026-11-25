"""Section 5 and Appendix D: the simulated sign of the self-consistency proxy.

For every (problem, model) pair of the organizers' olympiad sample, k = 8
answers are drawn: the correct one with the pair's measured base accuracy,
otherwise one of m distinct wrong answers uniformly.  Agreement is the share
of matching sample pairs, as in solutions/self-consistency.  The AUC of
"high agreement -> robust" is averaged over replicates; the AUC of the
opposite sign is one minus it.  m is assumed, not measured: this is a
simulation conditional on m, under both label rules.

The last block is the perturbation alternative of Appendix D: the rule "zero
decay under one perturbation type -> robust", per type, on the pairs that have
that type, and the Spearman correlation of decays between types.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from aimo_reference import pairs  # noqa: E402

K = 8
REPS = 4000
RNG = np.random.default_rng(20260918)
full = pd.read_parquet(ROOT / "data-hf" / "aimo-interp-challenge-sample-full.parquet")


def agreement(base, m):
    n = base.shape[0]
    correct = RNG.random((REPS, n, K)) < base[None, :, None]
    wrong = RNG.integers(0, m, size=(REPS, n, K))
    sq = correct.sum(axis=2).astype(float) ** 2
    for j in range(m):
        sq += ((~correct) & (wrong == j)).sum(axis=2).astype(float) ** 2
    return (sq - K) / (K * (K - 1))


for label in ("zero", "official"):
    fr = pairs(full, label)
    y = fr.is_robust.to_numpy().astype(int)
    base = fr.base.to_numpy()
    print(f"\n=== label rule {label}: {len(fr)} pairs, {y.sum()} robust, k = {K}, {REPS} replicates ===")
    print(f"{'m':>3} {'AUC high agr -> robust':>23} {'AUC low agr -> robust':>22} {'sd over reps':>13}")
    for m in (1, 2, 4, 8):
        a = agreement(base, m)
        aucs = np.array([roc_auc_score(y, a[r]) for r in range(REPS)])
        print(f"{m:>3} {aucs.mean():>23.3f} {1 - aucs.mean():>22.3f} {aucs.std():>13.3f}")

print("\n=== perturbation alternative (Appendix D) ===")
print("predictor: decay of one type == 0 (zero label) or <= 0.10 (official); label recomputed from")
print("the OTHER types only, so the predictor is not part of its own label")
per_type = full.groupby(["model_id", "original_problem", "permutation_type"]).absolute_accuracy_decay.max().unstack()
cols = ["typos", "distract", "rephrase", "rename", "domain"]
for t in cols:
    sub = per_type[per_type[t].notna()]
    others = [c for c in per_type.columns if c != t]
    y0 = sub[others].apply(lambda r: bool((r.dropna() == 0).all()), axis=1).astype(int)
    md = sub[others].max(axis=1)
    keep = (md <= 0.10) | (md >= 0.25)
    a0 = roc_auc_score(y0, (sub[t] == 0).astype(int))
    a1 = roc_auc_score((md[keep] <= 0.10).astype(int), (sub[t][keep] <= 0.10).astype(int))
    print(f"  {t:<9} zero: pairs={len(sub):>2} AUC={a0:.3f}   official: pairs={int(keep.sum()):>2} AUC={a1:.3f}")
rho = per_type[cols].corr(method="spearman")
off = rho.where(~np.eye(len(cols), dtype=bool)).stack()
print(f"  Spearman between types: min {off.min():.2f}, median {off.median():.2f}, max {off.max():.2f}")
print(rho.round(2).to_string())
