from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from reasoning_core.evaluation.metrics import (
    EvalExample,
    contrastive_mc_loss,
    contrastive_mc_objective_id,
)
from reasoning_core.evaluation.gradient import (
    GradientCache,
    GradientCacheSpec,
    build_eval_gradient_cache,
    completion_loss,
    gradient_objective_id,
    load_gradient_cache,
    score_task,
    score_task_gradient,
)


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "~"

    def __call__(self, text, add_special_tokens=True):
        ids = ([2] if add_special_tokens else []) + [3 + ord(char) % 29 for char in text]
        return SimpleNamespace(input_ids=ids)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.embed = nn.Embedding(32, 8)
        self.head = nn.Linear(8, 32, bias=False)

    def forward(self, input_ids, labels, attention_mask=None):
        logits = self.head(self.embed(input_ids))
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1), ignore_index=-100,
        )
        return SimpleNamespace(loss=loss, logits=logits)


@pytest.fixture
def tiny():
    return TinyLM(), CharTokenizer()


def test_completion_loss_matches_trl_completion_only_collator(tiny):
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    model, tokenizer = tiny
    rows = [
        {"prompt": "Q:a? A:", "completion": " x~"},
        {"prompt": "Q:bb? A:", "completion": " yz~"},
    ]
    actual = completion_loss(model, tokenizer, rows, max_length=32)
    examples = []
    for row in rows:
        prompt_ids = tokenizer(row["prompt"]).input_ids
        input_ids = tokenizer(row["prompt"] + row["completion"]).input_ids
        examples.append({
            "input_ids": input_ids,
            "completion_mask": [0] * len(prompt_ids) + [1] * (len(input_ids) - len(prompt_ids)),
        })
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id, completion_only_loss=True,
    )
    batch = collator(examples)
    expected = model(**batch).loss
    assert actual.loss.item() == pytest.approx(expected.item(), rel=1e-6)
    assert actual.examples == 2
    assert actual.tokens == int((batch["labels"][:, 1:] != -100).sum())


def test_completion_loss_reports_skipped_rows(tiny):
    model, tokenizer = tiny
    result = completion_loss(model, tokenizer, [
        {"prompt": "p", "completion": "a"},
        {"prompt": "far too long", "completion": "also too long"},
    ], max_length=5)
    assert result.examples == 1
    assert result.skipped_examples == 1


def test_contrastive_mc_loss_is_differentiable_and_content_addressed(tiny):
    model, tokenizer = tiny
    examples = (EvalExample("p", "a", ("a", "bc", "d"), 0),)
    result = contrastive_mc_loss(model, tokenizer, examples, 16, temperature=0.7)
    result.loss.backward()
    assert result.examples == 1
    assert result.tokens == 4
    assert any(parameter.grad is not None for parameter in model.parameters())
    first = contrastive_mc_objective_id("arc", examples, 16, 0.7)
    assert first != contrastive_mc_objective_id("arc", examples, 16, 1.0)
    assert first != contrastive_mc_objective_id("arc", examples, 15, 0.7)


def test_cache_round_trip_and_task_scoring(tiny, tmp_path):
    model, tokenizer = tiny
    legs = {
        "easy": (EvalExample("p", "a", ("a", "b"), 0),),
        "hard": (EvalExample("q", "b", ("a", "b"), 1),),
    }
    spec = GradientCacheSpec(
        "warm:test", gradient_objective_id(legs, 16, 1.0), 16, "float32",
    )
    built = build_eval_gradient_cache(
        model, tokenizer, legs, spec, cache_dir=tmp_path, model_id="tiny",
    )
    assert built.norm == pytest.approx(1.0, rel=1e-6)
    loaded = load_gradient_cache(spec, device="cpu", cache_dir=tmp_path)
    result = score_task_gradient(
        model, tokenizer, [{"prompt": "p", "completion": "a"}], loaded, 16,
    )
    assert -1 <= result.cosine <= 1
    assert result.eval_norm == pytest.approx(1.0, rel=1e-6)
    manifest = next(tmp_path.glob("*/manifest.json")).read_text()
    assert '"model": "tiny"' in manifest
    assert '"parameters"' in manifest


def test_multi_batch_score_uses_n_backwards_and_accumulated_gradient(tiny):
    model, tokenizer = tiny
    rows = [
        [{"prompt": "p", "completion": "a"}],
        [{"prompt": "q", "completion": "bc"}],
    ]
    model.zero_grad(set_to_none=True)
    completion_loss(model, tokenizer, rows[0], 16).loss.backward()
    gradients = {name: parameter.grad.detach().clone()
                 for name, parameter in model.named_parameters()}
    norm = sum(value.square().sum().item() for value in gradients.values()) ** 0.5
    cache = GradientCache(
        GradientCacheSpec("warm:test", "objective:test", 16, "float32"),
        gradients, norm,
    )
    estimate = score_task(model, tokenizer, rows, cache, batches=2, max_length=16)
    assert estimate.batches[0].cosine == pytest.approx(1.0, abs=1e-6)
    assert estimate.stderr_cosine == pytest.approx(estimate.std_cosine / 2**0.5)
    assert -1 <= estimate.aggregate_cosine <= 1
