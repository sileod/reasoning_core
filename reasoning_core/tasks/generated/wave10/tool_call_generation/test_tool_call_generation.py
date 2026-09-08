import random

from reasoning_core.template import Entry
from reasoning_core.tasks.generated.wave10.tool_call_generation.tool_call_generation import (
    ToolCallGeneration,
    ToolCallGenerationV1Config,
    _compute_total,
    _parse_call,
)


def test_config_difficulty_changes():
    cfg = ToolCallGenerationV1Config()
    l0 = cfg.offered
    cfg.set_level(3)
    assert cfg.offered > l0
    cfg.set_level(0)
    assert cfg.offered == 2
    assert cfg.allow_double is False
    assert cfg.allow_tax is False


def test_generate_and_score_gold():
    random.seed(12345)
    task = ToolCallGeneration()
    for _ in range(20):
        ex = task.generate_example()
        assert isinstance(ex, Entry)
        assert ex.prompt
        assert task.score_answer(ex.answer, ex) == 1.0


def test_score_wrong_and_junk():
    random.seed(999)
    task = ToolCallGeneration()
    ex = task.generate_example()
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("zzz", ex) == 0.0
    assert task.score_answer("reajrjrje9595!", ex) == 0.0
    assert task.score_answer("nope(1,2)", ex) == 0.0


def test_score_tolerates_spacing_and_case():
    random.seed(7)
    task = ToolCallGeneration()
    ex = task.generate_example()
    relaxed = ex.answer.replace("=", " = ").replace(",", " , ")
    assert task.score_answer(relaxed, ex) == 1.0
    lower = ex.answer.lower()
    assert task.score_answer(lower, ex) == 1.0


def test_wrong_value_scores_zero():
    random.seed(42)
    task = ToolCallGeneration()
    ex = task.generate_example()
    name, params = _parse_call(ex.answer)
    for k in params:
        gold = ex.answer.split("(", 1)[1][:-1]
        pieces = [p.strip() for p in gold.split(",")]
        for i, pc in enumerate(pieces):
            pk, pv = pc.split("=", 1)
            if pk.strip() == k:
                break
        else:
            continue
        wrong = _wrong_value(_parse_call(ex.answer)[1][k])
        pieces[i] = f"{pk.strip()}={wrong}"
        candidate = name + "(" + ", ".join(pieces) + ")"
        assert task.score_answer(candidate, ex) == 0.0, (k, candidate)


def test_all_levels_generate():
    random.seed(5)
    task = ToolCallGeneration()
    for level in range(7):
        ex = task.generate_example(level=level)
        assert task.score_answer(ex.answer, ex) == 1.0


def test_answer_variance():
    random.seed(11)
    task = ToolCallGeneration()
    seen = set()
    for _ in range(30):
        ex = task.generate_example()
        seen.add(ex.answer)
    assert len(seen) > 10


def _wrong_value(canon):
    if canon.startswith("i:"):
        return str(int(canon[2:]) + 1)
    if canon in ("true", "false"):
        return "false" if canon == "true" else "true"
    return canon + "_x"
