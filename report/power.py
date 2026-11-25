"""Power of the AIMO leaderboards, for the rule we actually submitted.

Deployed rule (solutions/self-consistency, decide()): inside each model group,
robust <=> score >= median of the group (np.median semantics, ties robust).
With a continuous score a group of g cases yields ceil(g/2) robust calls, so a
4/4/3/3 Small test always gets s = 8 of 14 and a 5/4/4/4/4 Main test s = 11
of 21; with the discrete agreement of 8 samples ties push s higher still
(runnoise.py).  Model groups are assigned to cases independently of the
labels.  The batch-median rule of the original report is kept as a reference.

"Declared better" means the score exceeds the critical value of a one-sided
alpha = 0.05 test whose null is the same rule at AUC 0.5, i.e. s robust calls
on a random subset of cases, which is exactly hypergeometric; the non-randomized
test is conservative because the score is discrete, the randomized one
attains alpha exactly.
"""
import numpy as np
from scipy.stats import norm, hypergeom

RNG = np.random.default_rng(20260919)
SMALL_GROUPS = (4, 4, 3, 3)
MAIN_GROUPS = (5, 4, 4, 4, 4)


def binormal_mu(auc):
    return np.sqrt(2.0) * norm.ppf(auc)


def groups_for(n, k):
    base, extra = divmod(n, k)
    return tuple([base + 1] * extra + [base] * (k - extra))


def simulate(n_pos, n_neg, auc, n_iter=200000, rule="group", groups=None):
    n = n_pos + n_neg
    mu = binormal_mu(auc)
    pos = RNG.normal(mu, 1.0, size=(n_iter, n_pos))
    neg = RNG.normal(0.0, 1.0, size=(n_iter, n_neg))
    s = np.concatenate([pos, neg], axis=1)
    y = np.zeros(n, dtype=bool)
    y[:n_pos] = True
    if rule == "median":
        pred = s >= np.median(s, axis=1, keepdims=True)
    elif rule == "group":
        perm = np.argsort(RNG.random((n_iter, n)), axis=1)
        sp = np.take_along_axis(s, perm, axis=1)
        yp = y[perm]
        pred = np.zeros_like(sp, dtype=bool)
        start = 0
        for g in groups:
            blk = sp[:, start:start + g]
            pred[:, start:start + g] = blk >= np.median(blk, axis=1, keepdims=True)
            start += g
        return (pred == yp).sum(axis=1)
    return (pred == y).sum(axis=1)


def n_robust_calls(n, groups=None):
    """Robust calls of a continuous score: ceil(g/2) per group, or the batch median."""
    if groups is None:
        return (n + 1) // 2
    return sum((g + 1) // 2 for g in groups)


def exact_null(n, n_pos, s):
    """P(correct = c) when s robust calls fall on a random subset: the null is exact."""
    ts = np.arange(0, min(s, n_pos) + 1)
    pmf = hypergeom.pmf(ts, n, n_pos, s)
    out = np.zeros(n + 1)
    np.add.at(out, n - n_pos - s + 2 * ts, pmf)
    return out


def critical(null, alpha=0.05):
    """Smallest c with P0(score > c) <= alpha, and gamma for the randomized test.
    null is either the exact pmf over 0..n or an array of simulated scores."""
    if null.ndim == 1 and null.dtype.kind == "f" and abs(null.sum() - 1) < 1e-9:
        pmf = null
    else:
        pmf = np.bincount(null.astype(int), minlength=null.max() + 2) / len(null)
    tail = pmf[::-1].cumsum()[::-1]
    above = np.append(tail[1:], 0.0)
    c = int(np.argmax(above <= alpha))
    gamma = (alpha - above[c]) / pmf[c] if pmf[c] > 0 else 0.0
    return c, gamma, above[c]


def power(c1, c, gamma):
    return (c1 > c).mean(), (c1 > c).mean() + gamma * (c1 == c).mean()


def report(name, n_pos, n_neg, groups, aucs=(0.50, 0.60, 0.65, 0.70, 0.80, 0.90)):
    n = n_pos + n_neg
    base = max(n_pos, n_neg)
    print(f"\n=== {name}: n={n}, robust={n_pos}, best constant={base}/{n}, groups={groups} ===")
    s = n_robust_calls(n, groups)
    c, gamma, a_nr = critical(exact_null(n, n_pos, s))
    print(f"exact null (s={s} robust calls on a random subset): critical: reject if score > {c}/{n} "
          f"(non-rand. alpha={a_nr:.4f}; randomized gamma={gamma:.4f} at {c})")
    print(f"{'AUC':>5} {'E[acc]':>8} {'sd(cases)':>10} {'P(>const)':>10} {'P(>=const+3)':>13} "
          f"{'power non-rand':>15} {'power rand':>11}")
    for auc in aucs:
        x = simulate(n_pos, n_neg, auc, groups=groups)
        pn, pr = power(x, c, gamma)
        print(f"{auc:>5.2f} {x.mean() / n:>8.4f} {x.std():>10.2f} {(x > base).mean():>10.3f} "
              f"{(x >= base + 3).mean():>13.3f} {pn:>15.3f} {pr:>11.3f}")
    print("minimum AUC detected with 80% power (randomized test):")
    for auc in np.arange(0.60, 0.991, 0.01):
        x = simulate(n_pos, n_neg, auc, n_iter=100000, groups=groups)
        if power(x, c, gamma)[1] >= 0.80:
            print(f"  AUC = {auc:.2f}")
            break


def null_tails():
    print("\n=== Exact null tails of the three method submissions, by s = number of True ===")
    print("correct = n - n_pos - s + 2t, t ~ Hypergeom(n, n_pos, s); s below the rule's minimum or of")
    print("the wrong parity cannot produce the observed score and is skipped")

    def tail(n, n_pos, s, obs, upper):
        ts = np.arange(0, min(s, n_pos) + 1)
        pmf = hypergeom.pmf(ts, n, n_pos, s)
        correct = n - n_pos - s + 2 * ts
        if not np.any(correct == obs) or pmf[correct == obs].sum() == 0:
            return None
        return pmf[correct >= obs].sum() if upper else pmf[correct <= obs].sum()

    for sid, n, n_pos, obs, upper, s_min in (
        ("932631 Small 8/14", 14, 7, 8, True, sum((g + 1) // 2 for g in SMALL_GROUPS)),
        ("933231 Small 5/14", 14, 7, 5, False, sum((g + 1) // 2 for g in SMALL_GROUPS)),
        ("933057 Main 7/21", 21, 10, 7, False, sum((g + 1) // 2 for g in MAIN_GROUPS)),
    ):
        side = ">=" if upper else "<="
        rows = []
        for s in range(s_min, n + 1):
            p = tail(n, n_pos, s, obs, upper)
            if p is not None:
                rows.append(f"s={s}: {p:.3f}")
        print(f"{sid:<20} s_min={s_min}  P(correct {side} {obs}):  " + ", ".join(rows))


def main():
    report("Small track v8, deployed per-model median", 7, 7, SMALL_GROUPS)
    report("Main track v8, deployed per-model median", 10, 11, MAIN_GROUPS)

    print("\n=== Power against test size, AUC 0.65, alpha 0.05 ===")
    print("deployed rule with 4 equal model groups (Small-track layout); batch median for reference")
    print(f"{'n':>4} {'crit':>9} {'alpha':>6} {'pow non-rand':>13} {'pow rand':>9} {'| batch: crit':>14} {'pow rand':>9}")
    for n in (14, 21, 40, 60, 100, 150, 200, 300, 400):
        n_pos, n_neg = n // 2, n - n // 2
        g = groups_for(n, 4)
        c1 = simulate(n_pos, n_neg, 0.65, n_iter=100000, groups=g)
        c, gm, a_nr = critical(exact_null(n, n_pos, n_robust_calls(n, g)))
        pn, pr = power(c1, c, gm)
        b1 = simulate(n_pos, n_neg, 0.65, n_iter=100000, rule="median")
        cb, gb, _ = critical(exact_null(n, n_pos, n_robust_calls(n)))
        print(f"{n:>4} {c:>5}/{n:<3} {a_nr:>6.3f} {pn:>13.3f} {pr:>9.3f} "
              f"{cb:>8}/{n:<4} {power(b1, cb, gb)[1]:>9.3f}")

    print("\n=== Two independent entries of the same method, Small track, deployed rule ===")
    for auc in (0.50, 0.65):
        a = simulate(7, 7, auc, groups=SMALL_GROUPS)
        b = simulate(7, 7, auc, groups=SMALL_GROUPS)
        d = np.abs(a - b)
        print(f"AUC={auc:.2f}: E|delta| = {d.mean():.2f} cases, P(|delta| >= 3) = {(d >= 3).mean():.3f}")

    null_tails()


if __name__ == "__main__":
    main()
