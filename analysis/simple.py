import sys, numpy as np, pandas as pd
sys.path.insert(0, "analysis")
from features import feats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

RS = 20260909
agg = pd.read_csv("baselines/data/math-robust-agg.csv").drop_duplicates("problem_id").reset_index(drop=True)
y = agg.model_is_robust.astype(int).values
F = feats(agg.original_problem)
ds = agg.dataset_id.values

print("== univariate AUC, pooled (n=%d, robust %.3f) ==" % (len(y), y.mean()))
u = sorted(((roc_auc_score(y, F[c].values), c) for c in F.columns), key=lambda t: -abs(t[0] - 0.5))
for a, n in u[:8]:
    print(f"  {n:16s} pooled {a:.4f}   test100 {roc_auc_score(y[ds!='hendrycks-math-train-reference-500'], F[n].values[ds!='hendrycks-math-train-reference-500']):.4f}"
          f"   train500 {roc_auc_score(y[ds=='hendrycks-math-train-reference-500'], F[n].values[ds=='hendrycks-math-train-reference-500']):.4f}")

CAND = {
    "len1_logreg": ["n_words"],
    "len2_logreg": ["n_words", "mean_word_len"],
    "len3_logreg": ["n_words", "mean_word_len", "latex_ratio"],
    "all_logreg": list(F.columns),
}
def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0,
                         class_weight="balanced", random_state=RS))
def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                  random_state=RS, n_jobs=4)

N_REP = 20
store = {}
for name, cols in list(CAND.items()) + [("all_rf", list(F.columns))]:
    X = F[cols].values
    mk = rf if name == "all_rf" else lr
    oof = np.zeros((N_REP, len(y)))
    for rep in range(N_REP):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=RS + rep).split(X, y):
            m = mk(); m.fit(X[tr], y[tr]); oof[rep, te] = m.predict_proba(X[te])[:, 1]
    aucs = [roc_auc_score(y, oof[r]) for r in range(N_REP)]
    store[name] = oof
    # leave-one-dataset-out transfer
    tr_auc = []
    for d in np.unique(ds):
        tr, te = ds != d, ds == d
        m = mk(); m.fit(X[tr], y[tr])
        tr_auc.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    print(f"{name:14s} OOF AUC {np.mean(aucs):.4f} +- {np.std(aucs):.4f}   transfer AUC {tr_auc[0]:.4f}/{tr_auc[1]:.4f}")

rng = np.random.default_rng(RS)
it, jf = np.where(y == 1)[0], np.where(y == 0)[0]
print("\n== simulated private test (19 cases, 10 robust / 9 not) ==")
print(f"  {'always_true':14s} 10.00/19 = 0.5263")
for name, oof in store.items():
    a5, atop = [], []
    for rep in range(N_REP):
        s = oof[rep]
        for _ in range(1500):
            sel = np.concatenate([rng.choice(it, 10, replace=False), rng.choice(jf, 9, replace=False)])
            yy, ss = y[sel], s[sel]
            a5.append(accuracy_score(yy, (ss > 0.5).astype(int)))
            pr = np.zeros(19, int); pr[np.argsort(-ss)[:10]] = 1
            atop.append(accuracy_score(yy, pr))
    a5, atop = np.array(a5), np.array(atop)
    print(f"  {name:14s} thr0.5 {a5.mean()*19:5.2f}/19 P(>10)={(a5>10/19).mean():.3f}"
          f" | top10 {atop.mean()*19:5.2f}/19 P(>10)={(atop>10/19).mean():.3f}")
