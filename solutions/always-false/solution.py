"""Baseline that classifies every model/problem pair as non-robust."""


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    """Return one false robustness label per input problem.

    Args:
        model_id: Evaluated model identifier, unused by this constant baseline.
        reasoning_effort: Evaluated reasoning effort, unused by this constant baseline.
        problems: Mathematical problem statements in evaluation order.

    Returns:
        Native ``False`` values aligned with ``problems``.
    """

    del model_id
    return [False for _ in problems]
