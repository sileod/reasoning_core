import random
import re

from reasoning_core.tasks.generated.wave8.nominal_subtyping.nominal_subtyping import (
    NominalSubtyping, NominalSubtypingConfig,
)


def _make(level, seed):
    random.seed(seed)
    t = NominalSubtyping(config=NominalSubtypingConfig())
    t.config.set_level(level)
    return t


def test_generate_scores_gold():
    for level in (0, 2, 5):
        for seed in range(5):
            t = _make(level, seed * 7 + 1)
            ex = t.generate_example()
            assert t.score_answer(ex.answer, ex) == 1.0


def test_junk_scores_low():
    t = NominalSubtyping(config=NominalSubtypingConfig())
    random.seed(3)
    ex = t.generate_example()
    assert t.score_answer("", ex) < 1.0
    assert t.score_answer("garbage text here", ex) < 1.0


def test_positive_chain_is_witness():
    t = NominalSubtyping(config=NominalSubtypingConfig())
    for seed in range(20):
        random.seed(seed)
        ex = t.generate_example()
        if ex.answer.startswith("YES"):
            chain = ex.answer[len("YES:"):].split(",")
            chain = [c.strip() for c in chain]
            assert chain[0] == ex.metadata.a
            # every consecutive pair must be a direct inheritance
            h = ex.metadata.hierarchy
            for i in range(len(chain) - 1):
                assert chain[i + 1] in h[chain[i]]


def test_negative_witness_is_not_subtype():
    t = NominalSubtyping(config=NominalSubtypingConfig())
    for seed in range(20):
        random.seed(seed * 3)
        ex = t.generate_example()
        if ex.answer.startswith("NO"):
            # witness list must be exactly A's strict supertypes, none of which is B
            a, b = ex.metadata.a, ex.metadata.b
            h = ex.metadata.hierarchy
            seen = set()
            stack = list(h[a])
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                stack.extend(h[node])
            witness = [s for s in ex.answer[len("NO:"):].split(",")]
            witness = [s.strip() for s in witness if s.strip()]
            assert set(witness) == set(seen)
            assert b not in seen and a != b


def test_format_answer_positive():
    t = NominalSubtyping(config=NominalSubtypingConfig())
    random.seed(42)
    ex = t.generate_example()
    if ex.answer.startswith("YES"):
        p = t.render_prompt(ex.metadata)
        assert "Is %s a subtype of %s?" % (ex.metadata.a, ex.metadata.b) in p
