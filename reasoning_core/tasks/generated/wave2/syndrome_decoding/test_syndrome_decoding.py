import json

from reasoning_core.tasks.generated.wave2.s20_syndrome_decoding.syndrome_decoding import (
    SyndromeDecoding,
    SyndromeDecodingConfig,
)


def test_gold_scores_one():
    t = SyndromeDecoding()
    for level in (0, 1, 2, 5):
        t.config.set_level(level)
        for _ in range(20):
            e = t.generate_example()
            assert t.score_answer(e.answer, e) == 1.0


def test_wrong_answers_score_zero():
    t = SyndromeDecoding()
    for level in (0, 2, 5):
        t.config.set_level(level)
        for _ in range(10):
            e = t.generate_example()
            a = list(e.answer)
            a[0] = "1" if a[0] == "0" else "0"
            wrong = "".join(a)
            assert t.score_answer(wrong, e) == 0.0


def test_garbage_does_not_crash():
    t = SyndromeDecoding()
    e = t.generate_example()
    assert t.score_answer("abc", e) == 0.0
    assert t.score_answer("", e) == 0.0
    assert t.score_answer(None, e) == 0.0
    assert t.score_answer("0 1 0", e) == 0.0


def test_answer_is_binary_string():
    t = SyndromeDecoding()
    for level in (0, 2, 5):
        t.config.set_level(level)
        for _ in range(10):
            e = t.generate_example()
            assert all(ch in "01" for ch in e.answer)
            assert len(e.answer) == e.metadata.n


def test_received_differs_from_corrected_by_at_most_one():
    t = SyndromeDecoding()
    for level in (0, 2, 5):
        t.config.set_level(level)
        for _ in range(20):
            e = t.generate_example()
            diff = sum(a != b for a, b in zip(e.answer, e.metadata.received))
            assert diff <= 1
            assert (e.metadata.er is not None) == (diff == 1)


def test_corrected_is_in_kernel_of_checks():
    t = SyndromeDecoding()
    for level in (0, 2, 5):
        t.config.set_level(level)
        for _ in range(10):
            e = t.generate_example()
            word = [int(ch) for ch in e.answer]
            for rule in e.metadata.payload["rules"]:
                inner = rule.split("positions ", 1)[1].strip("[]").strip()
                positions = [] if not inner else [int(x) for x in inner.split(",")]
                parity = 0
                for p in positions:
                    parity ^= word[p - 1]
                assert parity == 0
            assert e.metadata.received in (e.answer, "".join(
                str(int(ch) ^ 1) if i == e.metadata.er else ch
                for i, ch in enumerate(e.answer)
            ))


def test_syndrome_located_corrupted_position():
    t = SyndromeDecoding()
    for level in (0, 2, 5):
        t.config.set_level(level)
        for _ in range(20):
            e = t.generate_example()
            rword = [int(ch) for ch in e.metadata.received]
            if e.metadata.er is not None:
                assert rword[e.metadata.er] != int(e.answer[e.metadata.er])


def test_metadata_json_serializable():
    t = SyndromeDecoding()
    t.config.set_level(2)
    for _ in range(10):
        e = t.generate_example()
        json.dumps(dict(e.metadata))


def test_config_difficulty_changes():
    c = SyndromeDecodingConfig()
    c.set_level(0)
    base = c.n_bits
    c.set_level(5)
    assert c.n_bits > base


def test_difficulty_monotonic():
    prev = None
    for level in range(6):
        c = SyndromeDecodingConfig()
        c.set_level(level)
        nb = c.n_bits
        if prev is not None:
            assert nb >= prev
        prev = nb
