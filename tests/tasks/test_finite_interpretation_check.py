import os

from reasoning_core.tasks._tptp_finite_interpretation import (
    FiniteInterpretation,
    constant_symbols,
    complete_model,
    eval_cnf_clause,
    model_is_nondegenerate,
    parse_vampire_model,
    requirements_hold,
    serialize_model,
    universally_quantify,
    write_signed_fmb_problem,
)
from reasoning_core.tasks.deprecated.math_tptp_dev import FiniteInterpretationCheck


VAMPIRE_MODEL = """
% SZS output start FiniteModel for example
tff('declare_$i1',type,a:$i).
tff('declare_$i2',type,b:$i).
tff('finite_domain_$i',axiom,
      ! [X:$i] : (
         X = a | X = b
      ) ).

tff('distinct_domain_$i',axiom,
         a != b
).

tff(declare_f,type,f: ($i) > $i).
tff(function_f,axiom,
           f(a) = b
         & f(b) = a

).

tff(declare_p,type,p: ($i) > $o).
tff(predicate_p,axiom,
           p(a)
         & ~p(b)

).

% SZS output end FiniteModel for example
"""


def test_formula_evaluator_supports_emitted_cnf_subset():
    model = FiniteInterpretation(
        domain=[0, 1],
        constants={"a": 0, "b": 1},
        functions={"f": {(0,): 1, (1,): 0}},
        predicates={"p": {(0,): True, (1,): False}},
    )

    assert eval_cnf_clause("p(a) & ~p(b)", model)
    assert eval_cnf_clause("f(f(X)) = X", model)
    assert eval_cnf_clause("p(X) | ~p(X)", model)
    assert not eval_cnf_clause("p(X)", model)


def test_parse_and_totalize_vampire_model():
    partial = parse_vampire_model(VAMPIRE_MODEL)
    requirements = [
        {"formula": "p(a)", "should_be": True},
        {"formula": "~p(b)", "should_be": True},
        {"formula": "f(f(X)) = X", "should_be": True},
        {"formula": "p(X)", "should_be": False},
    ]
    model = complete_model(requirements, partial)

    assert model is not None
    assert requirements_hold(requirements, model)
    assert set(model.functions["f"]) == {(0,), (1,)}
    assert set(model.predicates["p"]) == {(0,), (1,)}


def test_signed_problem_writer_and_serialization():
    requirements = [
        {"formula": "p(X)", "should_be": True},
        {"formula": "q(X) | r(a)", "should_be": False},
    ]
    path = write_signed_fmb_problem(requirements, min_domain_size=3)
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    finally:
        os.remove(path)

    assert "cnf(r_0, axiom, p(X))." in text
    assert "fof(r_1, axiom, ~(![X] : (q(X) | r(a))))." in text
    assert text.count("fic_distinct_") == 3
    assert universally_quantify("p(a)") == "p(a)"

    model_text = serialize_model(FiniteInterpretation(domain=[0]))
    assert "Domain:\n{0}" in model_text
    assert "Constants:\n(none)" in model_text


def test_model_quality_filter_and_sparse_serialization():
    constant = FiniteInterpretation(
        domain=[0, 1],
        functions={"f": {(0,): 0, (1,): 0}},
        predicates={"p": {(0,): False, (1,): False}},
    )
    assert constant_symbols(constant) == [("function", "f"), ("predicate", "p")]
    assert not model_is_nondegenerate(constant)
    assert not model_is_nondegenerate(FiniteInterpretation(domain=[0]))

    varied = FiniteInterpretation(
        domain=[0, 1],
        functions={"f": {(0,): 0, (1,): 1}},
        predicates={"p": {(0,): False, (1,): True}},
    )
    assert model_is_nondegenerate(varied)
    text = serialize_model(varied, sparse=True)
    assert "default ->" in text
    assert text.count("->") == 4


def test_negative_model_is_generated_by_sign_flip(monkeypatch):
    requirements = [
        {"formula": "p(a)", "should_be": True},
        {"formula": "p(b)", "should_be": True},
    ]
    model = FiniteInterpretation(
        domain=[0, 1],
        constants={"a": 0, "b": 1},
        predicates={"p": {(0,): False, (1,): True}},
    )
    captured = {}

    def fake_fmb(solver_requirements, **kwargs):
        captured["requirements"] = solver_requirements
        return model

    monkeypatch.setattr(
        "reasoning_core.tasks.deprecated.math_tptp_dev.run_vampire_fmb_signed",
        fake_fmb,
    )
    task = object.__new__(FiniteInterpretationCheck)
    task.config = type("Config", (), {
        "min_domain_size": 2,
        "max_domain_size": 2,
        "fmb_time_limit": "1",
        "allow_constant_symbols": 0,
    })()
    monkeypatch.setattr("random.randrange", lambda _: 0)

    generated_model, verdicts, flipped_index = task._generate_model(requirements, True)
    assert generated_model is model
    assert captured["requirements"][0]["should_be"] is False
    assert captured["requirements"][1]["should_be"] is True
    assert verdicts == [False, True]
    assert flipped_index == 0


def test_prompt_and_per_requirement_scoring():
    task = object.__new__(FiniteInterpretationCheck)
    prompt = task.prompt({
        "axiom_set": "GRP001-0.ax",
        "context_axioms": [],
        "requirements": [{"formula": "p(X)", "should_be": False}],
        "model": "Domain:\n{0, 1}",
    })

    assert "at least one variable assignment" in prompt
    assert "N: True" in prompt

    entry = type("Entry", (), {
        "answer": "1: False\n2: True",
        "metadata": {
            "requirements": [{}, {}],
            "verdicts": [False, True],
        },
    })()
    assert task.score_answer("2: true\n1: false", entry) == 1.0
    assert task.score_answer("1: true\n2: true", entry) == 0.0
