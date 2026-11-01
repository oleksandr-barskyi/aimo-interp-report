"""Run-to-run variability of the deployed method on a FIXED test.

Section 3 of the report needs two different numbers and the draft conflated
them:

  * across-test noise --- how much the score moves when both the cases and the
    sampled answers are redrawn.  ``power.py`` gives this one (E|delta| = 2.09
    cases, P(|delta| >= 3) = 0.265 under AUC 0.50).
  * within-test noise --- how much the score moves when the *same* 14 cases are
    scored twice by the same method, so the only random input is the sampling
    of k answers per problem at temperature 0.8.  This is the pair (8/14, 5/14)
    we actually observed.

The second is what this script estimates.  The generative model is the deployed
method itself: for a pair with base accuracy b, each of the k samples is the
correct answer with probability b and otherwise one of m distinct wrong answers
drawn uniformly; agreement is the probability that two random samples match,
   a = (sum_v c_v^2 - k) / (k (k - 1));
the prediction is ``robust <=> a <= median(batch) + 1e-9``, copied from
solutions/self-consistency/self_consistency.py.

Base accuracies and labels come from the organizers' own 80 olympiad pairs, so
the case distribution is measured rather than assumed; m is the one free
parameter, the same assumption section 5 states.  The fixed test is drawn as
7 robust + 7 not robust to match the measured composition of the v8 Small
track, and the model groups are assigned independently of the labels.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from aimo_reference import pairs  # noqa: E402

K = 8
N_TEST = 14
N_ROBUST = 7
N_TESTS = 3000
N_RUNS = 120
RNG = np.random.default_rng(20260922)


def agreement(base, m, n_runs, rng):
    """(n_runs, n_cases) agreement estimates for one fixed vector of base rates."""
    n = base.shape[0]
    correct = rng.random((n_runs, n, K)) < base[None, :, None]
    wrong_id = rng.integers(0, m, size=(n_runs, n, K))
    c_correct = correct.sum(axis=2)
    sq = c_correct.astype(np.float64) ** 2
    for j in range(m):
        c_j = ((~correct) & (wrong_id == j)).sum(axis=2)
        sq += c_j.astype(np.float64) ** 2
    return (sq - K) / (K * (K - 1))


def predict(a, groups=None):
    """robust <=> agreement <= median of its own batch (or of its model group)."""
    if groups is None:
        med = np.median(a, axis=1, keepdims=True)
        return a <= med + 1e-9
    out = np.zeros_like(a, dtype=bool)
    for g in np.unique(groups):
        sel = groups == g
        med = np.median(a[:, sel], axis=1, keepdims=True)
        out[:, sel] = a[:, sel] <= med + 1e-9
    return out


def main():
    full = pd.read_parquet(ROOT / "data-hf" / "aimo-interp-challenge-sample-full.parquet")
    fr = pairs(full)
    base_all = fr.base.to_numpy()
    y_all = fr.is_robust.to_numpy()
    pos = np.flatnonzero(y_all)
    neg = np.flatnonzero(~y_all)
    print(f"case population: {len(fr)} pairs, {len(pos)} robust, {len(neg)} not robust")
    print(f"fixed test drawn as {N_ROBUST} robust + {N_TEST - N_ROBUST} not robust, k = {K} samples")
    print(f"{N_TESTS} fixed tests x {N_RUNS} runs each\n")

    groups4 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    for rule_name, groups in (("batch median (n=14)", None),
                              ("per-model median (4/4/3/3)", groups4)):
        print(f"--- decision rule: {rule_name} ---")
        print(f"{'m':>3} {'E[score]':>9} {'sd within':>10} {'sd total':>9} "
              f"{'sd between':>11} {'E|delta|':>9} {'P(|d|>=3)':>10} {'P(|d|>=3|worst)':>16}")
        for m in (1, 2, 4, 8):
            means = np.empty(N_TESTS)
            within_var = np.empty(N_TESTS)
            d_abs = np.empty(N_TESTS)
            d_tail = np.empty(N_TESTS)
            for t in range(N_TESTS):
                idx = np.concatenate([
                    RNG.choice(pos, size=N_ROBUST, replace=False),
                    RNG.choice(neg, size=N_TEST - N_ROBUST, replace=False),
                ])
                RNG.shuffle(idx)
                b = base_all[idx]
                y = y_all[idx]
                a = agreement(b, m, N_RUNS, RNG)
                pred = predict(a, groups)
                score = (pred == y[None, :]).sum(axis=1)
                means[t] = score.mean()
                within_var[t] = score.var()
                half = N_RUNS // 2
                d = np.abs(score[:half].astype(int) - score[half:].astype(int))
                d_abs[t] = d.mean()
                d_tail[t] = (d >= 3).mean()
            sd_within = np.sqrt(within_var.mean())
            sd_between = means.std()
            sd_total = np.sqrt(sd_within ** 2 + sd_between ** 2)
            print(f"{m:>3} {means.mean():>9.2f} {sd_within:>10.2f} {sd_total:>9.2f} "
                  f"{sd_between:>11.2f} {d_abs.mean():>9.2f} {d_tail.mean():>10.3f} "
                  f"{np.quantile(d_tail, 0.95):>16.3f}")
        print()

    print("Reference (power.py, across-test: cases AND answers redrawn):")
    print("  AUC 0.50  E|delta| = 2.09  P(|delta| >= 3) = 0.265")
    print("  AUC 0.65  E|delta| = 2.06  P(|delta| >= 3) = 0.258")


if __name__ == "__main__":
    main()
