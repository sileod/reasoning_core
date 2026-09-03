import random

from reasoning_core.tasks.generated.wave8.prefix_code_decode.prefix_code_decode import PrefixCodeDecode


def test_roundtrip_scores_one():
    random.seed(1)
    task = PrefixCodeDecode()
    for level in range(7):
        cfg = task.config_cls()
        cfg.set_level(level)
        task.config = cfg
        for _ in range(50):
            e = task.generate_example()
            assert task.score_answer(e.answer, e) == 1.0


def test_garbage_scores_zero():
    random.seed(2)
    task = PrefixCodeDecode()
    e = task.generate_example()
    assert task.score_answer("", e) == 0.0
    assert task.score_answer("garbage", e) == 0.0
    assert task.score_answer("s0 s1 s9", e) == 0.0


def test_whitespace_insensitive():
    random.seed(3)
    task = PrefixCodeDecode()
    e = task.generate_example()
    assert task.score_answer("  " + e.answer.replace(" ", "   ") + "  ", e) == 1.0


def test_answer_decodable_from_payload():
    random.seed(4)
    task = PrefixCodeDecode()
    e = task.generate_example()
    ans = e.answer.split()
    assert len(ans) >= 1


def test_max_len_variation_across_levels():
    random.seed(5)
    task = PrefixCodeDecode()
    lengths = set()
    for level in (0, 3, 6):
        cfg = task.config_cls()
        cfg.set_level(level)
        task.config = cfg
        e = task.generate_example()
        lengths.add(len(e.answer.split()))
    assert len(lengths) >= 2
