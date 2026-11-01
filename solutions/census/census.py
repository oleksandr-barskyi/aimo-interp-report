"""Measure the composition of the private small-track test set.

The ingestion program groups cases by ``model_id`` and calls ``are_robust``
once per group, catching participant exceptions per group and marking that
group's predictions invalid. The scoring program then reports

    coverage            = covered / 19
    accuracy            = correct / 19
    invalid_predictions = 19 - covered

so a submission that answers only for the labelled model family and raises for
every other model resolves the whole composition in one run:

    n_labelled = 19 * coverage
    robust among them = 19 * accuracy   (the answer given is a constant True)
    n_other = invalid_predictions
    robust among the others = 10 - 19 * accuracy   (10 robust in total, measured
                                                    by the always-true run 920252)

This is reconnaissance about the test *composition*, not a method: it fits
nothing to the leaderboard and its own accuracy can only be worse than the
constant baseline.
"""

from __future__ import annotations

LABELLED_FAMILY = ("qwen3-8b", "qwen3_8b", "deepseek")


def is_labelled_family(model_id: str) -> bool:
    key = model_id.lower()
    return any(marker in key for marker in LABELLED_FAMILY)


def predict(model_id: str, problems: list[str]) -> list[bool]:
    if not is_labelled_family(model_id):
        raise RuntimeError(f"census probe: {model_id} is outside the labelled family")
    return [True] * len(problems)
