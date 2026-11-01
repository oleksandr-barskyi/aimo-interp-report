import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "analysis")
from features import feats
from sklearn.ensemble import RandomForestClassifier

RS = 20260909
df = pd.read_csv("baselines/data/math-robust-final.csv")
p = df.groupby("problem_id", sort=True).agg(y=("model_is_robust", "first"),
                                            text=("original_problem", "first")).reset_index()
y = p.y.astype(int).values
F = feats(p.text)
X = F.values

rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                            random_state=RS, n_jobs=4)
rf.fit(X, y)

trees = []
for est in rf.estimators_:
    t = est.tree_
    val = t.value.reshape(t.value.shape[0], -1)
    prob = val / np.clip(val.sum(axis=1, keepdims=True), 1e-12, None)
    trees.append({
        "l": t.children_left.tolist(),
        "r": t.children_right.tolist(),
        "f": t.feature.tolist(),
        "t": [round(float(v), 6) for v in t.threshold],
        "p": [round(float(v), 6) for v in prob[:, 1]],
    })
out = {
    "feature_names": list(F.columns),
    "n_train": int(len(y)),
    "train_robust_rate": float(y.mean()),
    "trees": trees,
}
with open("solutions/text-prior/forest.json", "w") as fh:
    json.dump(out, fh, separators=(",", ":"))

# self-check: pure-python inference must match sklearn
sys.path.insert(0, "solutions/text-prior")
import text_prior
mine = text_prior.scores(list(p.text))
ref = rf.predict_proba(X)[:, 1]
print("max abs diff pure-python vs sklearn:", float(np.max(np.abs(mine - ref))))
print("trees:", len(trees), "nodes:", sum(len(t["l"]) for t in trees))
