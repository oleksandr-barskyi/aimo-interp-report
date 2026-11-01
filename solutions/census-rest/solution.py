from census_rest import predict


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    return predict(model_id, problems)
