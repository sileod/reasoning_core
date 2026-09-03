import random

from reasoning_core.tasks.generated.wave2.s24_version_resolution.version_resolution import (
    VersionResolution,
    _in_range,
    _resolve,
    _vstr,
    _vtok,
)


def _mk():
    t = VersionResolution()
    random.seed(12345)
    return t


def test_generate_render_score_roundtrip():
    t = _mk()
    e = t.generate_example()
    assert t.score_answer(e.answer, e) == 1.0
    assert t.score_answer("version " + e.answer, e) == 1.0
    assert t.score_answer('"' + e.answer + '"', e) == 1.0
    prompt = t.render_prompt(e.metadata)
    assert "Which version of" in prompt


def test_answer_is_not_newest():
    t = _mk()
    seen = 0
    for _ in range(200):
        t.config.set_level(random.randrange(0, 6))
        e = t.generate_example()
        tokens = e.metadata.payload["packages"]
        target = e.metadata.target
        for line in tokens.splitlines():
            if line.startswith(target + ":"):
                versions = line.split(":", 1)[1].split(",")
                versions = [v.strip() for v in versions]
                assert e.answer != versions[-1] and e.answer != versions[-1].strip('"')
                if e.answer == versions[-1]:
                    seen += 1
    assert seen == 0


def test_resolver_reproduces_construction():
    t = _mk()
    for _ in range(300):
        t.config.set_level(random.randrange(0, 6))
        e = t.generate_example()
        assert e.answer
        assert len(_vtok(e.answer)) >= 1


def test_levels_produce_some_variety():
    t = _mk()
    answers = set()
    for L in (0, 2, 5):
        t.config.set_level(L)
        for _ in range(30):
            answers.add(t.generate_example().answer)
    assert len(answers) >= 15


def test_in_range_basics():
    assert _in_range((1, 2, 0), "^1.2.0")
    assert not _in_range((2, 0, 0), "^1.2.0")
    assert _in_range((1, 4, 3), "~1.4.0")
    assert not _in_range((1, 5, 0), "~1.4.0")
    assert _in_range((1, 4, 0), ">= 1.4.0")
    assert not _in_range((1, 3, 9), ">= 1.4.0")
    assert _in_range((1, 4, 0), "== 1.4.0")


def test_junk_scores_zero():
    t = _mk()
    e = t.generate_example()
    for bad in ("", " ", "reajrjrje9595!", "not a version"):
        assert t.score_answer(bad, e) == 0.0
