import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "analysis")
from features import feats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

RS = 20260909
agg = pd.read_csv("baselines/data/math-robust-agg.csv").drop_duplicates("problem_id").reset_index(drop=True)
y = agg.model_is_robust.astype(int).values
nw = feats(agg.original_problem).n_words.values.astype(float)
ds = agg.dataset_id.values

def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0,
                         class_weight="balanced", random_state=RS))

VAR = {"raw": nw.reshape(-1, 1), "log1p": np.log1p(nw).reshape(-1, 1)}
rng = np.random.default_rng(RS)
it, jf = np.where(y == 1)[0], np.where(y == 0)[0]
N_REP = 20
for name, X in VAR.items():
    oof = np.zeros((N_REP, len(y)))
    for rep in range(N_REP):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=RS + rep).split(X, y):
            m = lr(); m.fit(X[tr], y[tr]); oof[rep, te] = m.predict_proba(X[te])[:, 1]
    aucs = [roc_auc_score(y, oof[r]) for r in range(N_REP)]
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
    tr_auc = []
    for d in np.unique(ds):
        trm, tem = ds != d, ds == d
        m = lr(); m.fit(X[trm], y[trm]); tr_auc.append(roc_auc_score(y[tem], m.predict_proba(X[tem])[:, 1]))
    print(f"{name:6s} OOF AUC {np.mean(aucs):.4f}+-{np.std(aucs):.4f} transfer {tr_auc[0]:.4f}/{tr_auc[1]:.4f}"
          f" | thr0.5 {a5.mean()*19:5.2f}/19 P(>10)={(a5>10/19).mean():.3f} sd={a5.std()*19:.2f}"
          f" | top52.6% {atop.mean()*19:5.2f}/19 P(>10)={(atop>10/19).mean():.3f}")

X = VAR["log1p"]
m = lr(); m.fit(X, y)
sc, lg = m.named_steps["standardscaler"], m.named_steps["logisticregression"]
mean, scale = float(sc.mean_[0]), float(sc.scale_[0])
w, b = float(lg.coef_[0][0]), float(lg.intercept_[0])
cut = np.expm1(mean + scale * (-b / w))
print(f"\nfinal fit on all {len(y)}: log1p(n_words) coef {w:.4f} intercept {b:.4f}"
      f" -> decision boundary at n_words = {cut:.1f}")
pred_in = (m.predict_proba(X)[:, 1] > 0.5).astype(int)
print("in-sample: predicted robust", int(pred_in.sum()), "of", len(y), "acc", round(accuracy_score(y, pred_in), 4))
json.dump({"feature": "log1p_n_words", "mean": mean, "scale": scale, "coef": w, "intercept": b,
           "n_train": int(len(y)), "train_robust_rate": float(y.mean()),
           "word_cutoff": float(cut), "sklearn_free": True},
          open("solutions/text-prior/length_logreg.json", "w"), indent=1)
print("artifact written")
