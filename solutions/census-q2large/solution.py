from census_field import predict, saved_name

FAMILY = 'qwen2'


def question(config) -> bool:
    return int(getattr(config, "hidden_size", 0) or 0) >= 3584


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    return predict(model_id, problems, FAMILY, question)
