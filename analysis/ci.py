import sys, numpy as np, pandas as pd
sys.path.insert(0, "analysis")
from features import feats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RS = 20260909
agg = pd.read_csv("baselines/data/math-robust-agg.csv").drop_duplicates("problem_id").reset_index(drop=True)
y = agg.model_is_robust.astype(int).values
F = feats(agg.original_problem)
nw = np.log1p(F.n_words.values.astype(float)).reshape(-1, 1)

def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0,
                         class_weight="balanced", random_state=RS))
def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                  random_state=RS, n_jobs=4)

N_REP = 20
def oof_scores(X, mk):
    o = np.zeros((N_REP, len(y)))
    for rep in range(N_REP):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=RS + rep).split(X, y):
            m = mk(); m.fit(X[tr], y[tr]); o[rep, te] = m.predict_proba(X[te])[:, 1]
    return o.mean(0)

s_len = oof_scores(nw, lr)
s_rf = oof_scores(F.values, rf)

rng = np.random.default_rng(RS)
B = 5000
n = len(y)
d, al, ar = [], [], []
for _ in range(B):
    idx = rng.integers(0, n, n)
    if len(np.unique(y[idx])) < 2:
        continue
    a1 = roc_auc_score(y[idx], s_len[idx]); a2 = roc_auc_score(y[idx], s_rf[idx])
    al.append(a1); ar.append(a2); d.append(a1 - a2)
al, ar, d = np.array(al), np.array(ar), np.array(d)
def ci(a):
    return np.percentile(a, 2.5), np.percentile(a, 97.5)
print(f"n={n}, positives={int(y.sum())}, bootstrap B={len(d)} (resample problems)")
print(f"  length logreg  OOF AUC {roc_auc_score(y, s_len):.3f}  95% CI [{ci(al)[0]:.3f}, {ci(al)[1]:.3f}]")
print(f"  RF 34 features OOF AUC {roc_auc_score(y, s_rf):.3f}  95% CI [{ci(ar)[0]:.3f}, {ci(ar)[1]:.3f}]")
print(f"  difference     {d.mean():+.3f}  95% CI [{ci(d)[0]:+.3f}, {ci(d)[1]:+.3f}]  P(length>RF)={(d>0).mean():.3f}")
