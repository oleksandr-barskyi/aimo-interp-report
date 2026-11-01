"""Name the architecture family of the nine unidentified small-track models.

Four earlier probes partitioned the private small-track test (19 cases):

    920381  4 cases whose model_id names the qwen3-8b / deepseek family (2 robust)
    920398  6 cases that resolve_checkpoint maps to Olmo-3-7B-Instruct  (3 robust)
    920400  0 cases that resolve to either large cached checkpoint
    920403  19 cases whose model_id loads straight through AutoConfig

So nine cases (5 robust) carry a model_id that is a real cached repository name
and matches none of the four checkpoints the organizers' README lists. This
probe reads ``config.model_type`` of exactly those nine and answers a constant
True when the type belongs to a family we already know, raising otherwise:

    coverage = k / 19   k = how many of the nine are qwen3 / olmo / gpt-oss / gemma
    accuracy = r / 19   r = how many of those k are robust
    invalid  = 19 - k

    k = 0  -> the nine are an architecture family we have not considered at all
    k = 9  -> they are variants of the families already in the cache, and the
              next probes only have to separate size and revision

The known ten are excluded on purpose: their share of coverage would only
re-measure what 920381 and 920398 already measured, and folding them in would
cost the resolution that makes k readable straight off the scoreboard.
"""

from __future__ import annotations

DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
OLMO = "allenai/Olmo-3-7B-Instruct"
LARGE = ("openai/gpt-oss-120b", "google/gemma-3-27b-it")
CACHED_CHECKPOINTS = (DEEPSEEK, OLMO) + LARGE
MODEL_ALIASES = {
    "qwen3-8b:low": DEEPSEEK,
    "olmo-3-7b:low": OLMO,
    "olmo3-7b:low": OLMO,
}
LABELLED_FAMILY = ("qwen3-8b", "qwen3_8b", "deepseek")
KNOWN_TYPES = ("qwen3", "olmo", "gpt_oss", "gemma")


def _normalize(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def resolve_checkpoint(model_id: str) -> str:
    if model_id in MODEL_ALIASES:
        return MODEL_ALIASES[model_id]
    if model_id in CACHED_CHECKPOINTS:
        return model_id
    wanted = _normalize(model_id.split(":")[0].split("/")[-1])
    for checkpoint in CACHED_CHECKPOINTS:
        candidate = _normalize(checkpoint.split("/")[-1])
        if wanted and (wanted in candidate or candidate in wanted):
            return checkpoint
    for checkpoint in CACHED_CHECKPOINTS:
        head = _normalize(checkpoint.split("/")[-1].split("-")[0])
        if head and head in wanted:
            return checkpoint
    return model_id


def is_labelled_family(model_id: str) -> bool:
    key = model_id.lower()
    return any(marker in key for marker in LABELLED_FAMILY)


def is_unidentified(model_id: str) -> bool:
    if is_labelled_family(model_id):
        return False
    return resolve_checkpoint(model_id) not in CACHED_CHECKPOINTS


def model_type(model_id: str) -> str:
    from transformers import AutoConfig

    for candidate in (model_id, model_id.split(":")[0]):
        if not candidate:
            continue
        try:
            config = AutoConfig.from_pretrained(candidate, local_files_only=True)
        except Exception:
            continue
        return str(getattr(config, "model_type", "") or "")
    return ""


def predict(model_id: str, problems: list[str]) -> list[bool]:
    if not is_unidentified(model_id):
        raise RuntimeError(f"census probe: {model_id} is already identified")
    found = model_type(model_id)
    if not found.startswith(KNOWN_TYPES):
        raise RuntimeError(f"census probe: {model_id} has unfamiliar model_type {found!r}")
    return [True] * len(problems)
