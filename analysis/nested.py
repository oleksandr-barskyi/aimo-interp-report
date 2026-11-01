import sys, numpy as np, pandas as pd
sys.path.insert(0, "analysis")
from features import feats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score

RS = 20260909
df = pd.read_csv("baselines/data/math-robust-final.csv")
p = df.groupby("problem_id", sort=True).agg(y=("model_is_robust", "first"),
                                            text=("original_problem", "first")).reset_index()
y = p.y.astype(int).values
X = feats(p.text).values

CAND = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced", random_state=RS)),
    "gb": lambda: GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.05, random_state=RS),
    "rf": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=RS, n_jobs=4),
    "et": lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=RS, n_jobs=4),
}

N_REP = 10
oof = np.zeros((N_REP, len(y)))
picked = []
for rep in range(N_REP):
    outer = StratifiedKFold(5, shuffle=True, random_state=RS + rep)
    for tr, te in outer.split(X, y):
        best, bname = -1, None
        inner = StratifiedKFold(5, shuffle=True, random_state=RS + 1000 + rep)
        for name, mk in CAND.items():
            s = cross_val_predict(mk(), X[tr], y[tr], cv=inner, method="predict_proba")[:, 1]
            a = roc_auc_score(y[tr], s)
            if a > best:
                best, bname = a, name
        picked.append(bname)
        m = CAND[bname]() ; m.fit(X[tr], y[tr])
        oof[rep, te] = m.predict_proba(X[te])[:, 1]

aucs = [roc_auc_score(y, oof[r]) for r in range(N_REP)]
print(f"NESTED CV (model choice inside fold): AUC {np.mean(aucs):.4f} +- {np.std(aucs):.4f}")
print("picked:", pd.Series(picked).value_counts().to_dict())
np.save("analysis/oof_nested.npy", oof)

# --- simulation of the real test: draw 19 cases, 10 robust / 9 not, rank within batch
rng = np.random.default_rng(RS)
idx_t = np.where(y == 1)[0]; idx_f = np.where(y == 0)[0]
accs, base = [], []
for rep in range(N_REP):
    s = oof[rep]
    for _ in range(2000):
        sel = np.concatenate([rng.choice(idx_t, 10, replace=False), rng.choice(idx_f, 9, replace=False)])
        yy, ss = y[sel], s[sel]
        order = np.argsort(-ss)
        pred = np.zeros(19, int); pred[order[:10]] = 1
        accs.append(accuracy_score(yy, pred))
        base.append(accuracy_score(yy, np.ones(19, int)))
accs = np.array(accs)
print(f"\nSIMULATED small track (19 cases, 10/9, top-10 rule):")
print(f"  our accuracy   mean {accs.mean():.4f}  sd {accs.std():.4f}  = {accs.mean()*19:.2f}/19")
print(f"  always-True    {np.mean(base):.4f} = 10/19")
print(f"  P(beat 10/19) = {(accs>10/19).mean():.3f}   P(tie) = {(accs==10/19).mean():.3f}   P(worse) = {(accs<10/19).mean():.3f}")
for k in range(6, 20):
    pr = (np.round(accs*19).astype(int) == k).mean()
    if pr > 0.001: print(f"    {k}/19: {pr:.3f}")
