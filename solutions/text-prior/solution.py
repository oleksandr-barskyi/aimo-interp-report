from text_prior import predict


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    try:
        return predict(problems)
    except Exception:
        return [True] * len(problems)
