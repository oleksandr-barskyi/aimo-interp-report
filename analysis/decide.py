import sys, numpy as np, pandas as pd
sys.path.insert(0, "analysis")
from features import feats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

RS = 20260909
N_REP = 20
df = pd.read_csv("baselines/data/math-robust-final.csv")
p = df.groupby("problem_id", sort=True).agg(y=("model_is_robust", "first"),
                                            text=("original_problem", "first")).reset_index()
y = p.y.astype(int).values
X = feats(p.text).values

def mk():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                  class_weight="balanced", random_state=RS, n_jobs=4)

oof = np.zeros((N_REP, len(y)))
for rep in range(N_REP):
    cv = StratifiedKFold(5, shuffle=True, random_state=RS + rep)
    for tr, te in cv.split(X, y):
        m = mk(); m.fit(X[tr], y[tr])
        oof[rep, te] = m.predict_proba(X[te])[:, 1]
np.save("analysis/oof_rf.npy", oof)
np.save("analysis/y.npy", y)
aucs = [roc_auc_score(y, oof[r]) for r in range(N_REP)]
print(f"RF OOF AUC {np.mean(aucs):.4f} +- {np.std(aucs):.4f}  (n={len(y)}, robust rate {y.mean():.4f})")

rng = np.random.default_rng(RS)
it, jf = np.where(y == 1)[0], np.where(y == 0)[0]
RULES = ["always_true", "always_false", "thr0.5", "top10of19"]
acc = {k: [] for k in RULES}
for rep in range(N_REP):
    s = oof[rep]
    for _ in range(2000):
        sel = np.concatenate([rng.choice(it, 10, replace=False), rng.choice(jf, 9, replace=False)])
        yy, ss = y[sel], s[sel]
        acc["always_true"].append(accuracy_score(yy, np.ones(19, int)))
        acc["always_false"].append(accuracy_score(yy, np.zeros(19, int)))
        acc["thr0.5"].append(accuracy_score(yy, (ss > 0.5).astype(int)))
        pr = np.zeros(19, int); pr[np.argsort(-ss)[:10]] = 1
        acc["top10of19"].append(accuracy_score(yy, pr))
print(f"\nsimulated private test (19 cases, 10 robust / 9 not), {N_REP*2000} draws")
for k in RULES:
    a = np.array(acc[k])
    beat = (a > 10 / 19).mean(); tie = (a == 10 / 19).mean()
    print(f"  {k:12s} acc {a.mean():.4f} sd {a.std():.4f} = {a.mean()*19:5.2f}/19"
          f"   P(>10/19)={beat:.3f} P(=)={tie:.3f} P(<)={1-beat-tie:.3f}")
