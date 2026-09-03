import random

from reasoning_core.tasks.generated.wave8.overload_resolution.overload_resolution import (
    OverloadResolution,
    most_specific,
    resolve,
    applicable,
    signature_cost,
)


def test_deterministic_under_seed():
    t = OverloadResolution()
    random.seed(1234)
    e1 = t.generate_example()
    random.seed(1234)
    t2 = OverloadResolution()
    e2 = t2.generate_example()
    assert e1.metadata.payload == e2.metadata.payload
    assert e1.answer == e2.answer


def test_score_gold():
    t = OverloadResolution()
    x = t.generate_example()
    assert t.score_answer(x.answer, x) == 1.0


def test_score_junk():
    t = OverloadResolution()
    x = t.generate_example()
    assert t.score_answer("", x) == 0.0
    assert t.score_answer("garbage", x) == 0.0


def test_resolve_unique():
    overloads = [("int",), ("float",)]
    assert applicable(("int",), ("int",))
    assert most_specific(("int",), ("float",)) == 1
    assert resolve(overloads, ("int",)) == 0


def test_resolve_float_num():
    overloads = [("int",), ("float",), ("number",)]
    assert resolve(overloads, ("int",)) == 0
    assert resolve(overloads, ("float",)) == 1


def test_no_applicable():
    overloads = [("str",)]
    assert resolve(overloads, ("int",)) is None


def test_all_difficulty_levels():
    t = OverloadResolution()
    for level in range(7):
        t.config.set_level(level)
        x = t.generate_example()
        assert t.score_answer(x.answer, x) == 1.0


def test_render_prompt_mentions_format():
    t = OverloadResolution()
    x = t.generate_example()
    prompt = t.render_prompt(x.metadata)
    assert "Ambiguous" in prompt
    assert "comma-separated" in prompt


def test_metadata_json_roundtrip():
    import json
    t = OverloadResolution()
    x = t.generate_example()
    d = json.loads(json.dumps(x.metadata.payload))
    assert [list(s) for s in d["overloads"]] == [list(s) for s in x.metadata.payload["overloads"]]
    assert list(d["arg_types"]) == list(x.metadata.payload["arg_types"])


def test_answer_diversity():
    random.seed(99)
    t = OverloadResolution()
    t.config.set_level(4)
    answers = set()
    for _ in range(40):
        answers.add(t.generate_example().answer)
    assert len(answers) >= 3
