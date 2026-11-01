import numpy as np
from math import comb
from scipy.stats import norm, hypergeom

RNG = np.random.default_rng(20260919)

def binormal_mu(auc):
    return np.sqrt(2.0) * norm.ppf(auc)

def simulate(n_pos, n_neg, auc, n_iter=200000, rule="median"):
    n = n_pos + n_neg
    mu = binormal_mu(auc)
    pos = RNG.normal(mu, 1.0, size=(n_iter, n_pos))
    neg = RNG.normal(0.0, 1.0, size=(n_iter, n_neg))
    s = np.concatenate([pos, neg], axis=1)
    y = np.zeros(n, dtype=bool)
    y[:n_pos] = True
    if rule == "median":
        thr = np.median(s, axis=1, keepdims=True)
        pred = s >= thr
    elif rule == "topk":
        k = n_pos
        order = np.argsort(-s, axis=1)
        pred = np.zeros_like(s, dtype=bool)
        np.put_along_axis(pred, order[:, :k], True, axis=1)
    correct = (pred == y).sum(axis=1)
    return correct

def report(name, n_pos, n_neg, aucs=(0.50, 0.60, 0.65, 0.70, 0.80, 0.90)):
    n = n_pos + n_neg
    base = max(n_pos, n_neg)
    print(f"\n=== {name}: n={n}, robust={n_pos}, spurious={n_neg}, best constant={base}/{n}={base/n:.4f} ===")
    print(f"{'AUC':>5} {'E[acc]':>8} {'sd(cases)':>10} {'P(>const)':>10} {'P(>=const)':>11} {'P(>=const+3)':>13}")
    for auc in aucs:
        c = simulate(n_pos, n_neg, auc)
        print(f"{auc:>5.2f} {c.mean()/n:>8.4f} {c.std():>10.2f} "
              f"{(c > base).mean():>10.3f} {(c >= base).mean():>11.3f} {(c >= base+3).mean():>13.3f}")

def main():
    report("Small track v8 (2026-09-16)", 7, 7)
    report("Main track v8", 10, 11)
    report("Small track v7", 10, 9)

    print("\n=== How many cases are needed to separate AUC 0.65 from a coin? ===")
    print("Criterion: one-sided test of 'accuracy of the median rule > best constant',")
    print("alpha = 0.05 under AUC = 0.50, power = 0.80 under AUC = 0.65.")
    for n in (14, 21, 40, 60, 100, 150, 200, 300, 400, 600, 800):
        n_pos = n // 2
        n_neg = n - n_pos
        c0 = simulate(n_pos, n_neg, 0.50, n_iter=100000)
        c1 = simulate(n_pos, n_neg, 0.65, n_iter=100000)
        crit = np.quantile(c0, 0.95)
        alpha = (c0 > crit).mean()
        power = (c1 > crit).mean()
        print(f"n={n:>4}  crit={crit:>6.1f}/{n}  actual alpha={alpha:.3f}  power(AUC=0.65)={power:.3f}")

    print("\n=== Two runs of the same method, small track: 8/14 and 5/14 ===")
    print("If both runs draw from the same distribution, how large is the gap by chance?")
    for auc in (0.50, 0.65):
        a = simulate(7, 7, auc, n_iter=200000)
        b = simulate(7, 7, auc, n_iter=200000)
        d = np.abs(a - b)
        print(f"AUC={auc:.2f}: E|delta| = {d.mean():.2f} cases, P(|delta| >= 3) = {(d >= 3).mean():.3f}")

    print("\n=== Null distribution of the observed submissions (exact, hypergeometric) ===")
    def null_tail(n, n_pos, s, obs):
        # predictions: s True; t of them truly robust ~ Hypergeom(n, n_pos, s)
        # correct = t + (n - n_pos - (s - t)) = n - n_pos - s + 2t
        ts = np.arange(0, min(s, n_pos) + 1)
        pmf = hypergeom.pmf(ts, n, n_pos, s)
        correct = n - n_pos - s + 2 * ts
        return (pmf[correct <= obs].sum(), (pmf * correct).sum())
    for (n, n_pos, s, obs, label) in [
        (21, 10, 10, 7, "main 933057 (7/21), assuming s=10 True"),
        (21, 10, 11, 7, "main 933057 (7/21), assuming s=11 True"),
        (14, 7, 6, 5, "small 933231 (5/14), assuming s=6 True"),
        (14, 7, 8, 5, "small 933231 (5/14), assuming s=8 True"),
        (14, 7, 7, 8, "small 932631 (8/14), assuming s=7 True"),
    ]:
        p, e = null_tail(n, n_pos, s, obs)
        print(f"{label:<46} E[correct|null]={e:.2f}  P(correct<={obs}|null)={p:.3f}")


if __name__ == "__main__":
    main()
