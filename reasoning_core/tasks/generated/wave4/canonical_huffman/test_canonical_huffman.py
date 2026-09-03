from reasoning_core.tasks.generated.wave4.s40_canonical_huffman.canonical_huffman import (
    CanonicalHuffman,
)


def test_gold_scoring():
    task = CanonicalHuffman()
    for _ in range(50):
        e = task.generate_entry()
        assert task.score_answer(e.answer, e) == 1.0


def test_junk_scoring():
    task = CanonicalHuffman()
    e = task.generate_entry()
    assert task.score_answer("", e) == 0.0
    assert task.score_answer("xyz", e) == 0.0


def test_levels_produce_examples():
    task = CanonicalHuffman()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(5):
            e = task.generate_entry()
            assert e.answer
            assert task.score_answer(e.answer, e) == 1.0


def test_answer_formats_consistent():
    task = CanonicalHuffman()
    for _ in range(30):
        e = task.generate_entry()
        if e.metadata.mode == "length":
            assert e.answer.isdigit()
        else:
            assert set(e.answer) <= {"0", "1"}
