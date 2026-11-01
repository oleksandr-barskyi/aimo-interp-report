from census_arch import predict

PREFIXES = ('llama', 'mistral', 'mixtral', 'phi', 'qwen2')
INVERT = True


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    return predict(model_id, problems, PREFIXES, INVERT)
