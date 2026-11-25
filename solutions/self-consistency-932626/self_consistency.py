"""Zero-shot self-consistency: rank problems by how much the judged model agrees
with itself, and call the least self-consistent half robust.

Measured basis (aimo-interp/aimo-interp-challenge-sample-full, 80 model-problem
pairs on AIMO 2025 reference problems):

    base_accuracy == 0   ->  16 robust / 4 not
    0 < base < 1         ->   0 robust / 27 not
    base_accuracy == 1   ->   9 robust / 24 not

AUC of -base_accuracy against the robust label is 0.695: the less a model solves
a problem, the more robust it is, because decay = base - permuted has nothing to
subtract from zero.

base_accuracy needs reference answers, which a submission never sees. The
label-free stand-in is the agreement of k sampled answers with each other. When
a model's wrong answers scatter -- the normal case on olympiad problems --
agreement tracks accuracy, so low agreement marks the problems the model cannot
solve. Simulation over the 80 pairs, drawing k answers per pair from the
measured base_accuracy: AUC of low agreement is 0.634 with 4 distinct wrong
answers and 0.670 with 8. The sign only flips if a model repeats one identical
wrong answer every time (AUC 0.255), which competition-level problems make
implausible.

Threshold: the median of the call's own batch. No leaderboard number enters the
rule, and a degenerate batch (all scores equal) falls back to all-True, the
measured constant floor.
"""

from __future__ import annotations

import os
import re
import time
from collections import Counter

START = time.monotonic()

TOTAL_BUDGET_SECONDS = 3150.0
SAMPLES_PER_PROBLEM = 8
MAX_SEQUENCES_PER_CALL = 32
MAX_PROMPT_TOKENS = 2048
MAX_NEW_TOKENS = 6144
FORCED_ANSWER_TOKENS = 24
TEMPERATURE = 0.8
TOP_P = 0.95

SYSTEM_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by "
    "step, then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does "
    "not have a solution."
)

ANSWER_RE = re.compile(r"ANSWER\s*[:=]\s*\$?\\?\s*(NaN|-?\d[\d,\s]*)", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]{1,40})\}")
INTEGER_RE = re.compile(r"-?\d+")

_state = {"problems_seen": 0}


def assumed_total_problems() -> int:
    directory = os.path.dirname(os.path.abspath(__file__))
    try:
        entries = os.listdir(directory)
    except OSError:
        return 21
    return 14 if any(name.casefold() == "small.txt" for name in entries) else 21


def remaining_seconds() -> float:
    return TOTAL_BUDGET_SECONDS - (time.monotonic() - START)


def call_budget(n_problems: int) -> float:
    left = remaining_seconds()
    total = assumed_total_problems()
    pending = max(n_problems, total - _state["problems_seen"])
    share = left * n_problems / pending
    return max(30.0, min(share, left * 0.6))


def load_model_and_tokenizer(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="auto", **kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, **kwargs)
        model.to("cpu")
    model.eval()
    return model, tokenizer


def _input_device(model):
    embeddings = model.get_input_embeddings()
    if embeddings is not None:
        return embeddings.weight.device
    return next(model.parameters()).device


def format_prompt(tokenizer, problem: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    for payload in (messages, messages[1:]):
        try:
            formatted = tokenizer.apply_chat_template(
                payload, tokenize=False, add_generation_prompt=True
            )
        except (AttributeError, TypeError, ValueError, KeyError):
            continue
        if isinstance(formatted, str):
            return formatted
    return SYSTEM_PROMPT + "\n\n" + problem


def extract_answer(text: str) -> str | None:
    """The answer the model actually declared, or ``None`` if it declared none.

    Only an explicit ``ANSWER:`` line or a ``\\boxed{}`` counts. A stray integer
    inside an unfinished reasoning trace is not an answer, and treating it as one
    would hide exactly the non-convergence the score is built on.
    """

    for source in (text.rsplit("</think>", 1)[-1], text):
        matches = ANSWER_RE.findall(source)
        if matches:
            raw = matches[-1]
            if raw.lower() == "nan":
                return "NaN"
            digits = re.sub(r"[,\s]", "", raw)
            if digits not in ("", "-"):
                return str(int(digits))
        for candidate in reversed(BOXED_RE.findall(source)):
            found = INTEGER_RE.findall(candidate.replace(",", ""))
            if found:
                return str(int(found[-1]))
    return None


def _forcing_suffix(text: str) -> str:
    if "<think>" in text and "</think>" not in text:
        return "\n</think>\n\nANSWER:"
    return "\n\nANSWER:"


def _generate(model, tokenizer, prompts, device, deadline, max_length, **overrides):
    import torch

    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_tensors="pt",
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    prompt_length = int(encoded["input_ids"].shape[1])
    options = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_time": max(20.0, deadline - time.monotonic()),
    }
    options.update(overrides)
    with torch.inference_mode():
        sequences = model.generate(**encoded, **options)
    generated = sequences[:, prompt_length:]
    clean = tokenizer.batch_decode(generated, skip_special_tokens=True)
    raw = tokenizer.batch_decode(generated, skip_special_tokens=False)
    pad_token = tokenizer.pad_token or ""
    eos_token = tokenizer.eos_token or ""
    trimmed = []
    for text in raw:
        for marker in (pad_token, eos_token):
            if marker:
                text = text.replace(marker, "")
        trimmed.append(text)
    del encoded, sequences, generated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return clean, trimmed


def _generate_with_retry(model, tokenizer, prompts, device, deadline, max_length, depth=0, **overrides):
    """Halve the batch and try again once the card refuses it.

    The worker is shared, so free VRAM is not the nominal 96 GB and a batch that
    fits in isolation may not fit at submission time.
    """

    try:
        return _generate(model, tokenizer, prompts, device, deadline, max_length, **overrides)
    except Exception:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if depth >= 3 or len(prompts) < 2 or deadline - time.monotonic() < 20.0:
            return [], []
        middle = len(prompts) // 2
        clean, raw = [], []
        for half in (prompts[:middle], prompts[middle:]):
            part_clean, part_raw = _generate_with_retry(
                model, tokenizer, half, device, deadline, max_length, depth + 1, **overrides
            )
            if len(part_clean) != len(half):
                return [], []
            clean.extend(part_clean)
            raw.extend(part_raw)
        return clean, raw


def sample_answers(model_id: str, problems: list[str], deadline: float) -> list[list[str | None]]:
    model, tokenizer = load_model_and_tokenizer(model_id)
    device = _input_device(model)

    prompts = []
    owners = []
    for index, problem in enumerate(problems):
        formatted = format_prompt(tokenizer, problem)
        for _ in range(SAMPLES_PER_PROBLEM):
            prompts.append(formatted)
            owners.append(index)

    chunks = [
        (start, prompts[start : start + MAX_SEQUENCES_PER_CALL])
        for start in range(0, len(prompts), MAX_SEQUENCES_PER_CALL)
    ]
    answers: list[list[str | None]] = [[] for _ in problems]

    for position, (start, chunk) in enumerate(chunks):
        left = deadline - time.monotonic()
        if left <= 20.0:
            break
        chunk_deadline = time.monotonic() + left / (len(chunks) - position)
        texts, raw_texts = _generate_with_retry(
            model, tokenizer, chunk, device, chunk_deadline, MAX_PROMPT_TOKENS
        )
        if len(texts) != len(chunk):
            continue

        pending = [
            offset for offset, text in enumerate(texts) if extract_answer(text) is None
        ]
        forced = {}
        if pending and deadline - time.monotonic() > 20.0:
            forcing_prompts = [
                chunk[offset] + raw_texts[offset] + _forcing_suffix(raw_texts[offset])
                for offset in pending
            ]
            try:
                forced_texts, _ = _generate_with_retry(
                    model,
                    tokenizer,
                    forcing_prompts,
                    device,
                    time.monotonic() + max(20.0, min(90.0, deadline - time.monotonic())),
                    MAX_PROMPT_TOKENS + MAX_NEW_TOKENS + 64,
                    max_new_tokens=FORCED_ANSWER_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            except Exception:
                forced_texts = []
            for offset, text in zip(pending, forced_texts):
                forced[offset] = "ANSWER:" + text

        for offset, text in enumerate(texts):
            final = forced.get(offset, text)
            answers[owners[start + offset]].append(extract_answer(final))

    return answers


def agreement(answers: list[str | None]) -> float | None:
    """Probability that two of the sampled answers coincide.

    ``None`` means the model produced nothing parseable in that sample; each
    such sample counts as its own distinct answer, which is exactly the
    "did not converge" evidence the score is meant to capture.
    """

    if len(answers) < 2:
        return None
    labels = [
        answer if answer is not None else "unparsed-%d" % index
        for index, answer in enumerate(answers)
    ]
    counts = Counter(labels).values()
    total = len(labels)
    return sum(count * (count - 1) for count in counts) / (total * (total - 1))


def decide(scores: list[float | None]) -> list[bool]:
    """Robust exactly for the least self-consistent half of the batch."""

    measured = sorted(score for score in scores if score is not None)
    if not measured:
        return [True] * len(scores)
    middle = len(measured) // 2
    if len(measured) % 2:
        median = measured[middle]
    else:
        median = (measured[middle - 1] + measured[middle]) / 2.0
    return [score is None or score <= median + 1e-9 for score in scores]


def predict(model_id: str, problems: list[str]) -> list[bool]:
    deadline = time.monotonic() + call_budget(len(problems))
    try:
        answers = sample_answers(model_id, problems, deadline)
        scores = [agreement(group) for group in answers]
        decision = decide(scores)
    except Exception:
        decision = [True] * len(problems)
    _state["problems_seen"] += len(problems)
    return [bool(value) for value in decision]
