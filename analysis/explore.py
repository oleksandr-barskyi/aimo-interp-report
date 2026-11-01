import pandas as pd, numpy as np
df = pd.read_csv("baselines/data/math-robust-final.csv")
print(df.shape, list(df.columns))
print("models:", df.model_id.unique())
print("datasets:", df.dataset_id.value_counts().to_dict())
print("perm types:", df.permutation_type.value_counts().to_dict())
print("perm source:", df.permutation_source.value_counts().to_dict())
print("robust:", df.model_is_robust.value_counts().to_dict())
print("uniq problems:", df.problem_id.nunique(), "uniq text:", df.original_problem.nunique())
g = df.groupby("problem_id").model_is_robust
print("problems with mixed label across rows:", (g.nunique()>1).sum(), "of", g.ngroups)
print("rows per problem:", df.groupby("problem_id").size().value_counts().to_dict())
print(df.groupby(["dataset_id"]).model_is_robust.agg(["mean","size"]))
print(df.groupby(["permutation_type"]).model_is_robust.agg(["mean","size"]))
print(df[["base_accuracy","permuted_accuracy","absolute_accuracy_decay","relative_accuracy_decay","n_base_predictions","n_permuted_predictions","n_detrimental_permutations"]].describe())
# relation label <-> decay
print(df.groupby("model_is_robust").absolute_accuracy_decay.describe())
agg = pd.read_csv("baselines/data/math-robust-agg.csv")
print("AGG", agg.shape, list(agg.columns))
