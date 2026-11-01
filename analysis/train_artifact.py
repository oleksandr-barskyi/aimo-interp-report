import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "analysis")
from features import feats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib, sklearn

RS = 20260909
df = pd.read_csv("baselines/data/math-robust-final.csv")
p = df.groupby("problem_id", sort=True).agg(y=("model_is_robust", "first"),
                                            text=("original_problem", "first")).reset_index()
y = p.y.astype(int).values
F = feats(p.text)
X = F.values
print("sklearn", sklearn.__version__, "rows", len(y), "features", list(F.columns))

rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                            random_state=RS, n_jobs=1)
rf.fit(X, y)
joblib.dump({"model": rf, "feature_names": list(F.columns), "sklearn": sklearn.__version__,
             "n_train": int(len(y)), "train_robust_rate": float(y.mean())},
            "solutions/text-prior/text_prior_rf.joblib", compress=3)

lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1,
                   class_weight="balanced", random_state=RS))
lr.fit(X, y)
sc = lr.named_steps["standardscaler"]; lg = lr.named_steps["logisticregression"]
json.dump({"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
           "coef": lg.coef_[0].tolist(), "intercept": float(lg.intercept_[0]),
           "feature_names": list(F.columns)},
          open("solutions/text-prior/fallback_logreg.json", "w"), indent=1)
print("artifacts written")
