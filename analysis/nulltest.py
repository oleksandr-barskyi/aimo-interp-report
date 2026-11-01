import sys, numpy as np, pandas as pd
sys.path.insert(0, "analysis")
from features import feats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

RS = 20260909
df = pd.read_csv("baselines/data/math-robust-final.csv")
p = df.groupby("problem_id", sort=True).agg(y=("model_is_robust", "first"),
                                            text=("original_problem", "first")).reset_index()
y = p.y.astype(int).values
X = feats(p.text).values
NAMES = list(feats(p.text).columns)

def mk():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                  class_weight="balanced", random_state=RS, n_jobs=4)

def oof_auc(yy, n_rep=5, seed=RS):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed)
    oof = np.zeros((n_rep, len(yy)))
    for i, (tr, te) in enumerate(cv.split(X, yy)):
        m = mk(); m.fit(X[tr], yy[tr])
        oof[i // 5, te] = m.predict_proba(X[te])[:, 1]
    return float(np.mean([roc_auc_score(yy, oof[r]) for r in range(n_rep)])), oof

real, oof = oof_auc(y, n_rep=10)
print(f"real OOF AUC = {real:.4f}")

rng = np.random.default_rng(RS)
null = []
for b in range(150):
    yp = rng.permutation(y)
    a, _ = oof_auc(yp, n_rep=2, seed=RS + b)
    null.append(a)
null = np.array(null)
pval = (np.sum(null >= real) + 1) / (len(null) + 1)
print(f"null AUC mean {null.mean():.4f} sd {null.std():.4f} q95 {np.quantile(null,0.95):.4f}")
print(f"permutation p-value = {pval:.4f}")

# operating points on the real OOF scores
s = oof.mean(axis=0)
print("\n-- accuracy at various thresholds, in-sample prior 0.263 --")
print(f"always-False acc = {1-y.mean():.4f}   always-True acc = {y.mean():.4f}")
print(f"balanced acc @0.5 = {balanced_accuracy_score(y, s>0.5):.4f}")
for q in [0.4737, 0.5, 0.5263, 0.7372]:
    thr = np.quantile(s, 1 - q)
    pred = (s > thr).astype(int)
    print(f"top {q:.2%} as robust -> acc {accuracy_score(y,pred):.4f} balacc {balanced_accuracy_score(y,pred):.4f}")

# per-feature univariate AUC
print("\n-- univariate AUC per feature --")
u = sorted(((roc_auc_score(y, X[:, i]), NAMES[i]) for i in range(X.shape[1])),
           key=lambda t: -abs(t[0] - 0.5))
for a, n in u[:12]:
    print(f"  {n:18s} {a:.4f}")
np.save("analysis/oof_scores.npy", s)
