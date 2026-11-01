import sys, numpy as np, pandas as pd
sys.path.insert(0, "analysis")
from features import feats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

RS = 20260909
agg = pd.read_csv("baselines/data/math-robust-agg.csv").drop_duplicates("problem_id").reset_index(drop=True)
agg["y"] = agg.model_is_robust.astype(int)
F = feats(agg.original_problem); X = F.values; y = agg.y.values
ds = agg.dataset_id.values
print("n =", len(y), "robust rate", round(y.mean(), 4))
print(agg.groupby("dataset_id").agg(n=("y", "size"), robust=("y", "mean")))
agg["n_words"] = F.n_words.values
print(agg.groupby("dataset_id").n_words.describe()[["count", "mean", "50%"]])
print("AUC of n_words for dataset_id (train500=1):",
      round(roc_auc_score((ds == "hendrycks-math-train-reference-500").astype(int), F.n_words.values), 4))
for d in np.unique(ds):
    m = ds == d
    print(f"  within {d}: n={m.sum()} robust={y[m].mean():.3f} AUC(n_words)={roc_auc_score(y[m], F.n_words.values[m]):.4f}")

def mk():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                  random_state=RS, n_jobs=4)

print("\n-- leave-one-dataset-out transfer --")
for d in np.unique(ds):
    tr, te = ds != d, ds == d
    m = mk(); m.fit(X[tr], y[tr]); s = m.predict_proba(X[te])[:, 1]
    print(f"  train on other, test on {d}: n={te.sum()} AUC={roc_auc_score(y[te], s):.4f} "
          f"acc@0.5={accuracy_score(y[te], (s>0.5).astype(int)):.4f} always-False={1-y[te].mean():.4f}")

print("\n-- within-dataset CV (dataset artifact removed) --")
for d in np.unique(ds):
    m_ = ds == d
    Xd, yd = X[m_], y[m_]
    aucs = []
    for rep in range(10):
        oof = np.zeros(len(yd))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=RS + rep).split(Xd, yd):
            mm = mk(); mm.fit(Xd[tr], yd[tr]); oof[te] = mm.predict_proba(Xd[te])[:, 1]
        aucs.append(roc_auc_score(yd, oof))
    print(f"  {d}: n={m_.sum()} robust={yd.mean():.3f} OOF AUC={np.mean(aucs):.4f} +- {np.std(aucs):.4f}")
