import random
import json
import pytest

from reasoning_core.tasks.generated.wave9.rate_limit_token_bucket.rate_limit_token_bucket import (
    RateLimitTokenBucket, simulate, final_balance,
)

LEVELS = [0, 1, 2, 3, 4, 5, 6]


def setup_module(module):
    random.seed(42)


def test_gold_roundtrip_all_levels():
    task = RateLimitTokenBucket()
    for level in LEVELS:
        for _ in range(20):
            entry = task.generate_example(level=level)
            assert task.score_answer(entry.answer, entry) == 1.0


def test_metadata_json_serializable():
    task = RateLimitTokenBucket()
    for level in LEVELS:
        for _ in range(20):
            entry = task.generate_example(level=level)
            json.dumps(entry.metadata.payload)
            json.dumps(dict(entry.metadata))


def test_junk_scores_zero():
    task = RateLimitTokenBucket()
    for level in [0, 3, 6]:
        for _ in range(10):
            entry = task.generate_example(level=level)
            assert task.score_answer("", entry) == 0.0
            assert task.score_answer("garbage", entry) == 0.0
            assert task.score_answer("a,b,c", entry) == 0.0


def test_balance_answer_is_int_domain():
    task = RateLimitTokenBucket()
    for level in LEVELS:
        for _ in range(30):
            entry = task.generate_example(level=level)
            if entry.metadata.mode == "balance":
                val = int(entry.answer)
                assert 0 <= val <= entry.metadata.capacity


def test_balance_self_consistent():
    task = RateLimitTokenBucket()
    for level in LEVELS:
        for _ in range(30):
            entry = task.generate_example(level=level)
            m = entry.metadata
            if m.mode == "balance":
                assert int(entry.answer) == final_balance(
                    m.capacity, m.refill_rate, m.refill_every, m.initial_tokens, m.requests
                )
            else:
                acc, _ = simulate(m.capacity, m.refill_rate, m.refill_every, m.initial_tokens, m.requests, m.rate)
                assert [int(t) for t in entry.answer.split(",") if t != ""] == acc


def test_prompt_mentions_answer_format():
    task = RateLimitTokenBucket()
    for level in LEVELS:
        for _ in range(5):
            entry = task.generate_example(level=level)
            prompt = task.render_prompt(entry.metadata)
            assert "answer" in prompt
