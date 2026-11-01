from census_arch import predict

PREFIXES = ('mistral', 'mixtral')
INVERT = False


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    return predict(model_id, problems, PREFIXES, INVERT)
