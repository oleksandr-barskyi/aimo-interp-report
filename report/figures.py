"""The two figures of the report.

fig1-power.pdf   power of the leaderboard test against the number of cases
fig2-base80.pdf  base accuracy by robustness label on the organizers' 80 pairs

Palette: blue ordinal ramp (fig 1) and blue/orange categorical slots 1-2
(fig 2), both checked with the data-viz validator in light mode.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(HERE))
from aimo_reference import pairs  # noqa: E402
from power import simulate  # noqa: E402

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
RAMP = {0.60: "#86b6ef", 0.65: "#3987e5", 0.70: "#256abf", 0.80: "#104281"}
CAT = {"not robust": "#2a78d6", "robust": "#eb6834"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 9.5,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(length=3, width=0.7)


def fig_power():
    ns = [14, 21, 30, 40, 60, 80, 100, 150, 200, 300, 400]
    aucs = [0.60, 0.65, 0.70, 0.80]
    cache = HERE / "fig1-power.csv"
    if cache.exists():
        tab = pd.read_csv(cache, index_col=0)
        power = {a: tab[f"{a:.2f}"].tolist() for a in aucs}
        ns = tab.index.tolist()
        print("power grid loaded from", cache.name)
        return _draw_power(ns, aucs, power)
    power = {a: [] for a in aucs}
    # The score is discrete, so a non-randomized test cannot hit alpha = 0.05
    # exactly and its power is not monotone in n (Table 2 shows the
    # conservative non-randomized values).  For the curve we use the
    # randomized test that attains alpha = 0.05 exactly: reject if c > crit,
    # and with probability gamma if c == crit.
    for n in ns:
        n_pos, n_neg = n // 2, n - n // 2
        c0 = simulate(n_pos, n_neg, 0.50, n_iter=400000)
        crit = int(np.quantile(c0, 0.95))
        while (c0 > crit).mean() > 0.05:
            crit += 1
        p_above = (c0 > crit).mean()
        p_at = (c0 == crit).mean()
        gamma = 0.0 if p_at == 0 else min(1.0, max(0.0, (0.05 - p_above) / p_at))
        for a in aucs:
            c1 = simulate(n_pos, n_neg, a, n_iter=400000)
            power[a].append((c1 > crit).mean() + gamma * (c1 == crit).mean())
        print(f"n={n:>4} crit={crit:>4d} gamma={gamma:.3f} alpha={p_above + gamma * p_at:.4f} " +
              " ".join(f"AUC{a:.2f}={power[a][-1]:.3f}" for a in aucs))

    tab = pd.DataFrame(power, index=ns)
    tab.index.name = "n"
    tab.columns = [f"{a:.2f}" for a in aucs]
    tab.to_csv(HERE / "fig1-power.csv", float_format="%.4f")
    print("\nfig1 table:\n" + tab.to_string(float_format="%.3f"))
    return _draw_power(ns, aucs, power)


def _draw_power(ns, aucs, power):
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    style(ax)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)

    ax.axhline(0.80, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    ax.text(400, 0.80, "conventional 80% power", ha="right", va="center",
            fontsize=7.5, color=MUTED, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))

    for n_mark, label in ((14, "Small track"), (21, "Main track")):
        ax.axvline(n_mark, color=MUTED, linewidth=0.7, linestyle=(0, (1, 2)), zorder=1)
        ax.text(n_mark * 1.04, 1.02, label, rotation=90, fontsize=7.3,
                color=INK2, va="top", ha="left", zorder=5)

    # direct label each curve where it crosses 0.5, so identity is never colour alone
    label_at = {0.80: 0.90, 0.70: 0.65, 0.65: 0.50, 0.60: 0.40}
    for a in aucs:
        ys = np.array(power[a])
        xs = np.array(ns, dtype=float)
        ax.plot(xs, ys, color=RAMP[a], linewidth=2.0, zorder=3,
                solid_capstyle="round", label=f"AUC {a:.2f}")
        ax.plot([14, 21], [ys[0], ys[1]], "o", color=RAMP[a],
                markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4)
        xc = float(np.interp(label_at[a], ys, xs))
        ax.annotate(f"{a:.2f}", xy=(xc, label_at[a]), xytext=(0, 7),
                    textcoords="offset points", color=RAMP[a], fontsize=8,
                    ha="center", va="bottom", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none"))

    ax.set_xscale("log")
    ax.set_xlim(12.3, 430)
    ax.set_xticks([14, 21, 40, 60, 100, 200, 400])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("private test size, cases (log scale)")
    ax.set_ylabel("probability the method is\ndeclared better than the constant")
    ax.set_title("Power of the leaderboard test against test size, one-sided, $\\alpha = 0.05$",
                 loc="left", pad=8)
    leg = ax.legend(loc="lower right", fontsize=7.8, handlelength=1.3,
                    labelspacing=0.35, borderpad=0.4,
                    bbox_to_anchor=(0.995, 0.02))
    leg.set_zorder(6)
    fig.subplots_adjust(left=0.135, right=0.975, top=0.88, bottom=0.155)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig1-power.{ext}", dpi=220)
    plt.close(fig)


def fig_base80():
    full = pd.read_parquet(ROOT / "data-hf" / "aimo-interp-challenge-sample-full.parquet")
    fr = pairs(full)
    b = (fr.base.to_numpy() * 10).round().astype(int)
    y = fr.is_robust.to_numpy()
    bins = np.arange(0, 11)
    n_not = np.array([((b == k) & ~y).sum() for k in bins])
    n_rob = np.array([((b == k) & y).sum() for k in bins])
    print("bin  not robust  robust")
    for k in bins:
        print(f"{k/10:.1f} {n_not[k]:>10} {n_rob[k]:>7}")

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    style(ax)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)

    x = bins / 10.0
    bw, off = 0.028, 0.0155
    ax.axvspan(0.05, 0.95, color="#f2f1ed", zorder=0)
    ax.bar(x - off, n_not, width=bw, color=CAT["not robust"],
           label="not robust (55 pairs)", zorder=3)
    ax.bar(x + off, n_rob, width=bw, color=CAT["robust"],
           label="robust (25 pairs)", zorder=3)

    for xi, v in ((x[0] - off, n_not[0]), (x[10] - off, n_not[10]),
                  (x[0] + off, n_rob[0]), (x[10] + off, n_rob[10])):
        if v:
            ax.text(xi, v + 0.5, str(v), ha="center", fontsize=7.5, color=INK2)

    ax.text(0.5, 14.0,
            "shaded band: intermediate base accuracy\n"
            "27 pairs, none of them robust",
            fontsize=7.8, color=INK2, ha="center", va="center", linespacing=1.4)

    ax.set_xlim(-0.06, 1.06)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}" for v in x])
    ax.set_ylim(0, 28)
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_xlabel("base accuracy of the (problem, model) pair, rounded to 0.1")
    ax.set_ylabel("pairs")
    ax.set_title("Robustness label against base accuracy, 80 olympiad pairs",
                 loc="left", pad=8)
    ax.legend(loc="upper center", ncol=2, fontsize=8, handlelength=1.1,
              handleheight=0.9, borderpad=0.2, columnspacing=1.6,
              bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.87, bottom=0.175)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig2-base80.{ext}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    fig_power()
    print()
    fig_base80()
