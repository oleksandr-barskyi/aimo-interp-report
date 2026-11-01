import sys, numpy as np, pandas as pd
sys.path.insert(0, "analysis")
from features import feats
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

RS = 20260909
df = pd.read_csv("baselines/data/math-robust-final.csv")
p = df.groupby("problem_id", sort=True).agg(y=("model_is_robust", "first"),
                                            text=("original_problem", "first")).reset_index()
y = p.y.astype(int).values
X = feats(p.text).values
texts = p.text.values
print(f"problems={len(y)} robust={y.sum()} rate={y.mean():.4f}")

def make_models():
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced", random_state=RS)),
        "logreg_C1": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced", random_state=RS)),
        "gb": GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.05, random_state=RS),
        "rf": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=RS, n_jobs=4),
        "et": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=RS, n_jobs=4),
    }

def run_cv(get_Xtr, name_models, n_repeats=20):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RS)
    scores = {k: [] for k in name_models}
    oof = {k: np.zeros((n_repeats, len(y))) for k in name_models}
    for i, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        rep = i // 5
        Xtr, Xte = get_Xtr(tr, te)
        for k, mk in name_models.items():
            m = mk()
            m.fit(Xtr, y[tr])
            s = m.predict_proba(Xte)[:, 1]
            oof[k][rep, te] = s
    out = {}
    for k in name_models:
        aucs = [roc_auc_score(y, oof[k][r]) for r in range(n_repeats)]
        out[k] = (np.mean(aucs), np.std(aucs), oof[k])
    return out

# ---- numeric features
def get_num(tr, te):
    return X[tr], X[te]
res = run_cv(get_num, {k: (lambda kk=k: make_models()[kk]) for k in make_models()})
print("\n== hand features, pooled OOF ROC AUC (5-fold x 20 repeats, group=problem) ==")
for k, (m, s, _) in sorted(res.items(), key=lambda x: -x[1][0]):
    print(f"{k:12s} AUC {m:.4f} +- {s:.4f}")

# ---- tfidf
def tf_models():
    return {"tfidf_lr": lambda: make_pipeline(LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced", random_state=RS))}
def get_tfidf(tr, te):
    v = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), stop_words="english")
    return v.fit_transform(texts[tr]), v.transform(texts[te])
res2 = run_cv(get_tfidf, tf_models())
for k, (m, s, _) in res2.items():
    print(f"{k:12s} AUC {m:.4f} +- {s:.4f}")

np.save("analysis/oof_logreg.npy", res["logreg"][2])
np.save("analysis/y.npy", y)
