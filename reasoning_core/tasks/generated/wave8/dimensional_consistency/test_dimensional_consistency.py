import ast

from reasoning_core.tasks.generated.wave8.dimensional_consistency.dimensional_consistency import (
    DimensionalConsistency,
    parse_answer,
)


def _compute(payload):
    # Re-derive dim(X) from the payload to sanity check the gold answer.
    # payload only carries the literal equation/var_desc text; instead verify
    # through the stored Xdim triple and the reused parse/score path.
    return payload


def test_gold_score_one():
    task = DimensionalConsistency()
    for level in (0, 2, 5):
        task.config.set_level(level)
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_prompt_mentions_answer_format():
    task = DimensionalConsistency()
    ex = task.generate_example()
    p = task.render_prompt(ex.metadata)
    assert "X" in p and "(m, l, t)" in p


def test_parse_answer():
    assert parse_answer("(1,2,-1)", None) == (1, 2, -1)
    assert parse_answer("[1, 2, -1]", None) == (1, 2, -1)
    assert parse_answer("garbage", None) is None


def test_junk_scores_zero():
    task = DimensionalConsistency()
    ex = task.generate_example()
    assert task.score_answer("", ex) == 0.0
    assert task.score_answer("not a triple", ex) == 0.0


def test_answer_never_zero_vector():
    task = DimensionalConsistency()
    for _ in range(60):
        task.config.set_level(3)
        ex = task.generate_example()
        t = ast.literal_eval(ex.answer)
        assert any(c != 0 for c in t)


def _parse_dim(s):
    return tuple(int(x) for x in s[1:-1].split(","))


def test_construction_is_dimensionally_valid():
    from reasoning_core.tasks.generated.wave8.dimensional_consistency.dimensional_consistency import (
        _add, _sub, _mul,
    )
    task = DimensionalConsistency()
    for level in (0, 2, 5, 6):
        task.config.set_level(level)
        for _ in range(20):
            ex = task.generate_example()
            md = ex.metadata
            dims = {k: _parse_dim(v) for k, v in md["vars"].items()}
            X = _parse_dim(md["Xdim"])
            # Re-parse the equation string and recompute every term's dimension.
            # Split on '=' then '+'.
            lhs, rhs = md["payload"]["equation"].split("=")
            terms = [t.strip() for t in lhs.split("+")] + [t.strip() for t in rhs.split("+")]
            dims_by_symbol = dict(dims)
            dims_by_symbol["X"] = X
            common = None
            for t in terms:
                body = t.split(" ", 1)[1] if " " in t else t
                v = [0, 0, 0]
                for part in body.split("*"):
                    part = part.strip()
                    if "^" in part:
                        name, pow_s = part.split("^")
                        p = int(pow_s)
                    else:
                        name, p = part, 1
                    v = _add(v, _mul(list(dims_by_symbol[name]), p))
                if common is None:
                    common = tuple(v)
                else:
                    assert tuple(v) == common, (md["payload"]["equation"], t, v, common)



def test_difficulty_changes_config():
    task = DimensionalConsistency()
    task.config.set_level(0)
    base = task.config.n_left
    task.config.set_level(6)
    assert task.config.n_left > base
