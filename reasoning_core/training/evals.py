"""Small, versioned evaluators for training and influence experiments."""

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import sys
import torch

DISCARD_WARN_RATE = 0.02   # warn once >2% of a leg's rows are unscorable


EVALUATOR_VERSION = 1


@dataclass(frozen=True)
class EvalExample:
    prompt: str
    answer: str
    choices: tuple[str, ...] = ()
    answer_index: int | None = None


@dataclass(frozen=True)
class EvalSuite:
    name: str
    examples: tuple[EvalExample, ...]

    @property
    def identifier(self):
        payload = {
            "version": EVALUATOR_VERSION,
            "name": self.name,
            "examples": [asdict(example) for example in self.examples],
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        return f"{self.name}/eval@v{EVALUATOR_VERSION}:{digest}"


def load_eval_suite(path, eos_token, name=None, limit=None):
    examples = []
    with Path(path).expanduser().open() as file:
        for line in file:
            row = json.loads(line)
            if row.get("prompt") is None or row.get("answer") is None:
                continue
            choices = tuple(str(choice) for choice in row.get("choices") or ())
            answer = str(row["answer"])
            answer_index = row.get("answer_idx")
            if choices and answer_index is None:
                answer_index = next(
                    (index for index, choice in enumerate(choices)
                     if choice.strip() == answer.strip()),
                    None,
                )
            prompt = f"{row['prompt']}\n" if row["prompt"] else ""
            examples.append(EvalExample(
                prompt=prompt,
                answer=f"{answer}{eos_token}",
                choices=choices,
                answer_index=int(answer_index) if answer_index is not None else None,
            ))
            if limit and len(examples) >= limit:
                break
    return EvalSuite(name or Path(path).stem, tuple(examples))


def load_qa_jsonl(path, eos_token, limit=None):
    suite = load_eval_suite(path, eos_token, limit=limit)
    return [(example.prompt, example.answer) for example in suite.examples]


@torch.no_grad()
def evaluate_qa_nll(model, tokenizer, examples, max_length):
    total_loss = total_tokens = 0
    per_example = []
    device = next(model.parameters()).device
    with evaluating(model):
        for item in examples:
            example = _example(item)
            prompt_ids = _prompt_ids(tokenizer, example.prompt)
            answer_ids = _token_ids(tokenizer, example.answer)
            if len(prompt_ids) + len(answer_ids) > max_length:
                per_example.append(None)
                continue
            input_ids = torch.tensor([prompt_ids + answer_ids], device=device)
            labels = input_ids.clone()
            labels[0, :len(prompt_ids)] = -100
            loss = model(input_ids, labels=labels).loss.item()
            tokens = (labels[0, 1:] != -100).sum().item()
            total_loss += loss * tokens
            total_tokens += tokens
            per_example.append(loss)
    if not total_tokens:
        raise RuntimeError("QA evaluation scored no answer tokens")
    return {
        "nll": total_loss / total_tokens,
        "tokens": total_tokens,
        "examples": len(examples),
        "scored_examples": sum(value is not None for value in per_example),
        "per_example": per_example,
    }


@torch.no_grad()
def evaluate_lm_nll(model, tokenizer, texts, max_length):
    total_loss = total_tokens = 0
    per_example = []
    device = next(model.parameters()).device
    with evaluating(model):
        for text in texts:
            ids = _token_ids(tokenizer, text)[:max_length]
            if len(ids) < 2:
                per_example.append(None)
                continue
            input_ids = torch.tensor([ids], device=device)
            loss = model(input_ids, labels=input_ids).loss.item()
            tokens = len(ids) - 1
            total_loss += loss * tokens
            total_tokens += tokens
            per_example.append(loss)
    if not total_tokens:
        raise RuntimeError("LM evaluation scored no tokens")
    return {
        "nll": total_loss / total_tokens,
        "tokens": total_tokens,
        "examples": len(texts),
        "scored_examples": sum(value is not None for value in per_example),
        "per_example": per_example,
    }


@torch.no_grad()
def evaluate_mcq(model, tokenizer, examples, max_length):
    """Choice accuracy plus gold NLL and gold-vs-best-distractor margin."""

    correct = total = 0
    gold_nlls, margins, per_example = [], [], []
    skipped = 0                      # rows dropped because prompt+candidate exceeded max_length
    device = next(model.parameters()).device
    with evaluating(model):
        for item in examples:
            example = _example(item)
            if not example.choices or example.answer_index is None:
                per_example.append(None)
                continue
            prompt_ids = _prompt_ids(tokenizer, example.prompt)
            scores = [
                _candidate_nll(model, tokenizer, prompt_ids, choice, max_length, device)
                for choice in example.choices
            ]
            gold = example.answer_index
            if not -len(scores) <= gold < len(scores) or all(
                score == float("inf") for score in scores
            ):
                skipped += 1
                per_example.append(None)
                continue
            if any(score == float("inf") for score in scores):
                skipped += 1     # partial: some candidates unscorable, margin is computed on the rest
            prediction = min(range(len(scores)), key=scores.__getitem__)
            distractors = [score for index, score in enumerate(scores)
                           if index != gold and score != float("inf")]
            margin = (
                min(distractors) - scores[gold]
                if distractors and scores[gold] != float("inf") else None
            )
            correct += prediction == gold
            total += 1
            if scores[gold] != float("inf"):
                gold_nlls.append(scores[gold])
            if margin is not None:
                margins.append(margin)
            per_example.append({"prediction": prediction, "gold_nll": scores[gold],
                                "margin": margin})
    # A row whose prompt+candidate exceeds max_length scores inf and is dropped SILENTLY. That is a
    # data-format bug (prompt too long for the battery's max_length), not a property of the model, and
    # it biases every metric below toward whichever rows happen to be short. Say so, loudly.
    if examples and skipped / len(examples) > DISCARD_WARN_RATE:
        print(f"[eval] WARNING: {skipped}/{len(examples)} rows ({skipped / len(examples):.1%}) exceeded "
              f"max_length={max_length} and were dropped. Shorten the prompts (cap so that "
              f"prompt+answer < max_length) or raise max_length; metrics below are computed on the "
              f"surviving, systematically shorter rows.", file=sys.stderr)
    return {
        "accuracy": correct / total if total else None,
        "examples": len(examples),
        "scored_examples": total,
        "discarded_examples": skipped,
        "gold_nll": sum(gold_nlls) / len(gold_nlls) if gold_nlls else None,
        "margin": sum(margins) / len(margins) if margins else None,
        "per_example": per_example,
    }


@torch.no_grad()
def evaluate_generation(model, tokenizer, examples, max_length, max_new_tokens,
                        score=None):
    """Greedy generation with an injectable task-specific scoring function."""

    score = score or _exact_match
    values, per_example = [], []
    device = next(model.parameters()).device
    with evaluating(model):
        for item in examples:
            example = _example(item)
            prompt_ids = _prompt_ids(tokenizer, example.prompt)
            if not prompt_ids or len(prompt_ids) > max_length:
                per_example.append(None)
                continue
            input_ids = torch.tensor([prompt_ids], device=device)
            output = model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            prediction = tokenizer.decode(output[0, len(prompt_ids):], skip_special_tokens=True)
            gold = example.answer
            if tokenizer.eos_token and gold.endswith(tokenizer.eos_token):
                gold = gold[:-len(tokenizer.eos_token)]
            value = float(score(prediction, gold))
            values.append(value)
            per_example.append({"prediction": prediction, "score": value})
    return {
        "score": sum(values) / len(values) if values else None,
        "scored_examples": len(values),
        "per_example": per_example,
    }


def eval_id(name, path, limit=None):
    content = Path(path).expanduser().read_bytes()
    digest = hashlib.sha256(content + f"\0limit={limit}".encode()).hexdigest()[:12]
    return f"{name}/answer_nll@v1:{digest}"


def save_eval(run_dir, identifier, result):
    path = Path(run_dir) / "evals" / f"{safe_name(identifier)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"eval_id": identifier, **result}, indent=2) + "\n")
    os.replace(tmp, path)
    return path


def load_eval(run_dir, identifier):
    path = Path(run_dir) / "evals" / f"{safe_name(identifier)}.json"
    return json.loads(path.read_text())


@contextmanager
def evaluating(model):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def safe_name(value):
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _example(value):
    if isinstance(value, EvalExample):
        return value
    prompt, answer = value
    return EvalExample(prompt, answer)


def _token_ids(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False)
    return encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]



def _prompt_ids(tokenizer, prompt):
    """Tokenize an eval prompt the way TRAINING formats it.

    data.py format_row (influence_auto_v1 / sft_qa_v1) emits prompt as f"{prompt}\n" and the
    completion as f"{answer}{eos}". The evaluators used to tokenize example.prompt raw, so the model
    was scored on a prefix it never saw in training -- a train/eval mismatch on every leg. Normalise
    trailing newlines so a leg whose stored prompt already ends in one is not double-terminated.
    """
    return _token_ids(tokenizer, str(prompt).rstrip("\n") + "\n")


def _candidate_nll(model, tokenizer, prompt_ids, candidate, max_length, device):
    answer_ids = _token_ids(tokenizer, str(candidate))
    if not answer_ids or len(prompt_ids) + len(answer_ids) > max_length:
        return float("inf")
    input_ids = torch.tensor([prompt_ids + answer_ids], device=device)
    labels = input_ids.clone()
    labels[0, :len(prompt_ids)] = -100
    return model(input_ids, labels=labels).loss.item()


def _exact_match(prediction, answer):
    normalize = lambda text: " ".join(str(text).strip().casefold().split())
    return normalize(prediction) == normalize(answer)
