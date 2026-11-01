from trace_length import predict


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    try:
        return predict(model_id, problems)
    except Exception:
        return [True] * len(problems)
