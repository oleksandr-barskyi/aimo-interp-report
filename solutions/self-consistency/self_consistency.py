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
FORCING_RESERVE = 0.15
TEMPERATURE = 0.8
TOP_P = 0.95

SYSTEM_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by "
    "step, then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does "
    "not have a solution."
)

ANSWER_RE = re.compile(r"ANSWER\s*[:=]", re.IGNORECASE)
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
    """Time this call may spend: its share of the problems still to come.

    The 0.6 ceiling stops one early group from eating the run when the assumed
    total is wrong, but it has nothing to protect once no problems remain after
    this call, and there it only burned 40% of the budget. Main track splits 21
    problems across five models against the small track's 14 across four, so the
    last group is the one that needs the whole remainder.
    """

    left = remaining_seconds()
    total = assumed_total_problems()
    pending = max(n_problems, total - _state["problems_seen"])
    share = left * n_problems / pending
    if pending > n_problems:
        share = min(share, left * 0.6)
    return max(30.0, min(share, left))


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


def _read_declared(source: str) -> str | None:
    position = -1
    for match in ANSWER_RE.finditer(source):
        position = match.end()
    if position < 0:
        return None
    line = source[position : position + 120].split("\n", 1)[0]
    if "nan" in line.lower():
        return "NaN"
    found = INTEGER_RE.findall(re.sub(r"[,\s]", "", line))
    return str(int(found[-1])) if found else None


def extract_answer(text: str) -> str | None:
    """The answer the model actually declared, or ``None`` if it declared none.

    Only an explicit ``ANSWER:`` line or a ``\\boxed{}`` counts. A stray integer
    inside an unfinished reasoning trace is not an answer, and treating it as one
    would hide exactly the non-convergence the score is built on.

    Within the declared line the *last* integer wins, so that "ANSWER: 4" and
    "ANSWER: 2 + 2 = 4" -- both shapes the forced continuation produces -- are
    counted as the same answer instead of as a disagreement.
    """

    for source in (text.rsplit("</think>", 1)[-1], text):
        declared = _read_declared(source)
        if declared is not None:
            return declared
        for candidate in reversed(BOXED_RE.findall(source)):
            found = INTEGER_RE.findall(candidate.replace(",", ""))
            if found:
                return str(int(found[-1]))
    return None


def _forcing_suffix(text: str) -> str:
    """Close an open reasoning block before demanding the answer.

    Two reasoning formats appear among the evaluated checkpoints.

    ``<think>`` tags: the tag is judged on prompt plus generation together,
    because the chat templates of Qwen3.5-4B, Skywork-OR1-Math-7B and
    Olmo-3-7B-Think already open ``<think>`` in the prompt, while
    DeepSeek-R1-0528-Qwen3-8B leaves the model to open it.

    Harmony channels: ``openai/gpt-oss-120b`` (main track only) has no
    ``<think>`` tag at all. Its prompt ends with ``<|start|>assistant`` and the
    model writes ``<|channel|>analysis<|message|>`` ... ``<|end|>``
    ``<|start|>assistant<|channel|>final<|message|>`` ... A trace cut inside the
    analysis channel must be closed by opening the final channel; appending a
    bare ``ANSWER:`` there would leave the demand inside the reasoning channel,
    the same mistake the ``<think>`` handling had before submission 932631.
    """

    if "<|start|>assistant" in text:
        if "<|channel|>final<|message|>" in text.rsplit("<|start|>assistant", 1)[-1]:
            return "\n\nANSWER:"
        return "<|end|><|start|>assistant<|channel|>final<|message|>ANSWER:"
    if "<think>" in text and "</think>" not in text:
        return "\n</think>\n\nANSWER:"
    return "\n\nANSWER:"


def _encode(tokenizer, prompts: list[str], max_length: int) -> list[list[int]]:
    return tokenizer(
        prompts, add_special_tokens=False, truncation=True, max_length=max_length
    )["input_ids"]


def _pad_left(id_lists: list[list[int]], pad_id: int, device):
    import torch

    width = max(len(row) for row in id_lists)
    ids = torch.full((len(id_lists), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(id_lists), width), dtype=torch.long)
    for index, row in enumerate(id_lists):
        if row:
            ids[index, width - len(row) :] = torch.tensor(row, dtype=torch.long)
            mask[index, width - len(row) :] = 1
    return ids.to(device), mask.to(device)


def _generate(model, tokenizer, id_lists, device, deadline, **overrides):
    """Generate from token ids and return token ids.

    Deliberately no decode-then-re-encode step: in transformers 5.13 the
    DeepSeek-R1-0528-Qwen3-8B tokenizer decodes its own ids without any
    whitespace ("ANSWER: 4" comes back as "ANSWER:4"), so feeding decoded text
    back in would hand the model a corrupted copy of its own reasoning.
    """

    import torch

    pad_id = tokenizer.pad_token_id
    ids, mask = _pad_left(id_lists, pad_id, device)
    prompt_width = int(ids.shape[1])
    options = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "use_cache": True,
        "pad_token_id": pad_id,
        "max_time": max(20.0, deadline - time.monotonic()),
    }
    options.update(overrides)
    with torch.inference_mode():
        sequences = model.generate(input_ids=ids, attention_mask=mask, **options)
    generated = []
    for row in sequences[:, prompt_width:].tolist():
        while row and row[-1] == pad_id:
            row.pop()
        generated.append(row)
    clean = tokenizer.batch_decode(generated, skip_special_tokens=True)
    raw = tokenizer.batch_decode(generated, skip_special_tokens=False)
    del ids, mask, sequences
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return generated, clean, raw


def _generate_with_retry(model, tokenizer, id_lists, device, deadline, depth=0, **overrides):
    """Halve the batch and try again once the card refuses it.

    The worker is shared, so free VRAM is not the nominal 96 GB and a batch that
    fits in isolation may not fit at submission time.
    """

    try:
        return _generate(model, tokenizer, id_lists, device, deadline, **overrides)
    except Exception:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if depth >= 3 or len(id_lists) < 2 or deadline - time.monotonic() < 20.0:
            return [], [], []
        middle = len(id_lists) // 2
        merged: tuple[list, list, list] = ([], [], [])
        for half in (id_lists[:middle], id_lists[middle:]):
            parts = _generate_with_retry(
                model, tokenizer, half, device, deadline, depth + 1, **overrides
            )
            if len(parts[0]) != len(half):
                return [], [], []
            for target, part in zip(merged, parts):
                target.extend(part)
        return merged


def sample_answers(model_id: str, problems: list[str], deadline: float) -> list[list[str | None]]:
    """Sample answers, keeping the forced read-out reachable under time pressure.

    Measured on submission 933057 (main track, 7/21): when the main generation
    ate the whole call budget, the forced pass was gated behind
    ``deadline - now > 20`` and simply did not run. Every truncated trace then
    stayed answer-less, every sample became its own label, agreement collapsed
    to 0 for the entire batch and the median rule degenerated to ``[True] * n``.
    Reproduced in ``temp/task-1059/diag3.py``: the pass fired at
    ``max_new_tokens = 32`` and never at 160 or 384.

    So the chunk's slice is split: ``1 - FORCING_RESERVE`` bounds the main
    generation, the rest belongs to the forced read-out, which is 24 greedy
    tokens and needs a fraction of what it is given. The slice as a whole is
    unchanged, so the run still fits the same 3150 seconds.
    """

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
        share = left / (len(chunks) - position)
        reserve = max(20.0, min(90.0, share * FORCING_RESERVE))
        chunk_deadline = time.monotonic() + max(20.0, share - reserve)
        prompt_ids = _encode(tokenizer, chunk, MAX_PROMPT_TOKENS)
        gen_ids, texts, raw_texts = _generate_with_retry(
            model, tokenizer, prompt_ids, device, chunk_deadline
        )
        if len(texts) != len(chunk):
            continue

        pending = [
            offset for offset, text in enumerate(texts) if extract_answer(text) is None
        ]
        forced = {}
        if pending:
            forcing_inputs = []
            for offset in pending:
                suffix = _forcing_suffix(chunk[offset] + raw_texts[offset])
                suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
                forcing_inputs.append(prompt_ids[offset] + gen_ids[offset] + suffix_ids)
            _, forced_texts, _ = _generate_with_retry(
                model,
                tokenizer,
                forcing_inputs,
                device,
                time.monotonic() + reserve,
                max_new_tokens=FORCED_ANSWER_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
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
