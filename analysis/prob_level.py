import pandas as pd, numpy as np, json
df = pd.read_csv("baselines/data/math-robust-final.csv")
ov = df[df.permutation_type=="OVERALL"]
print("OVERALL rows:", len(ov), "uniq pid:", ov.problem_id.nunique())
print("dup OVERALL pids:", ov.problem_id.value_counts().head())
p = df.groupby("problem_id").agg(label=("model_is_robust","first"), n=("model_is_robust","size"),
                                 text=("original_problem","first"),
                                 dec=("absolute_accuracy_decay","max"),
                                 ndet=("n_detrimental_permutations","sum"))
print("problems:", len(p), p.label.value_counts().to_dict(), "true rate", p.label.mean())
print("label vs max decay:", pd.crosstab(p.label, p.dec>0))
agg = pd.read_csv("baselines/data/math-robust-agg.csv")
print("agg uniq pid:", agg.problem_id.nunique(), "agg true rate", agg.model_is_robust.mean())
print("agg perm types:", agg.permutation_type.value_counts().to_dict())
# val-sample duplicates
import pathlib
lines=[json.loads(l) for l in open("getting-started/data/val-sample/input/cases.jsonl")]
labs={json.loads(l)["id"]:json.loads(l)["is_robust"] for l in open("getting-started/data/val-sample/reference/labels.jsonl")}
print("val cases:", len(lines), "uniq (model,problem):", len({(c["model_id"],c["problem"]) for c in lines}))
vs=pd.DataFrame(lines); vs["y"]=vs.id.map(labs)
print(vs.groupby("model_id").y.agg(["mean","size"]))
# overlap of val-sample problems with training csv
tr=set(df.original_problem)
print("val problems also in train csv:", sum(c["problem"] in tr for c in lines))
