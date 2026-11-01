from census_field import predict, saved_name

FAMILY = 'qwen3'


def question(config) -> bool:
    return True


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    return predict(model_id, problems, FAMILY, question)
