#!/usr/bin/env python3
"""What the aimo-2025-reference split says about robustness labels.

Runs on aimo-interp/aimo-interp-challenge-sample-full (data-hf/), the only
public split whose problems come from the AIMO 2025 reference set rather than
hendrycks MATH.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data-hf" / "aimo-interp-challenge-sample-full.parquet"
VAL_CASES = ROOT / "getting-started" / "data" / "val-sample" / "input" / "cases.jsonl"
VAL_LABELS = ROOT / "getting-started" / "data" / "val-sample" / "reference" / "labels.jsonl"


def pairs(full: pd.DataFrame) -> pd.DataFrame:
    grouped = full.groupby(["model_id", "original_problem"])
    frame = grouped.agg(
        is_robust=("absolute_accuracy_decay", lambda s: bool((s == 0).all())),
        base=("base_accuracy", "mean"),
        n_rows=("absolute_accuracy_decay", "size"),
    ).reset_index()
    frame["n_words"] = frame.original_problem.str.split().str.len()
    return frame


def check_label_rule(frame: pd.DataFrame) -> None:
    cases = pd.read_json(VAL_CASES, lines=True).rename(columns={"problem": "original_problem"})
    labels = pd.read_json(VAL_LABELS, lines=True)
    val = cases.merge(labels, on="id").merge(
        frame, on=["model_id", "original_problem"], how="left", suffixes=("", "_derived")
    )
    print(f"val-sample pairs matched in aimo-2025-reference: {val.n_rows.notna().sum()}/{len(val)}")
    print(f"label rule 'all(decay == 0)' agrees on: {(val.is_robust_derived == val.is_robust).mean():.3f}")


def main() -> None:
    full = pd.read_parquet(FULL)
    frame = pairs(full)
    print(f"rows {len(full)}, models {full.model_id.nunique()}, problems {full.problem_id.nunique()}")
    print(f"model-problem pairs {len(frame)}, robust {frame.is_robust.sum()}")
    print()
    check_label_rule(frame)
    print()
    print(pd.crosstab(frame.base.round(2), frame.is_robust).to_string())
    print()
    robust = frame[frame.is_robust]
    print(f"robust pairs with base_accuracy == 0: {(robust.base == 0).sum()}/{len(robust)}")
    print(f"robust pairs with base_accuracy == 1: {(robust.base == 1).sum()}/{len(robust)}")
    print(f"robust pairs with 0 < base < 1:       {((robust.base > 0) & (robust.base < 1)).sum()}/{len(robust)}")
    print()
    print(f"rule 'robust iff base == 0' accuracy: {((frame.base == 0) == frame.is_robust).mean():.3f}")
    print(f"AUC of (base == 0):                   {roc_auc_score(frame.is_robust, (frame.base == 0).astype(int)):.3f}")
    print(f"AUC of n_words, pooled:               {roc_auc_score(frame.is_robust, frame.n_words):.3f}")
    small = frame[frame.model_id == "qwen3-8b:low"]
    print(f"AUC of n_words, qwen3-8b only (n={len(small)}):  {roc_auc_score(small.is_robust, small.n_words):.3f}")


if __name__ == "__main__":
    main()
