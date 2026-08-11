import ast
import random
from types import SimpleNamespace

from reasoning_core.tasks.code_analysis import (
    CodeAnalysis,
    CodeAnalysisConfig,
    _BadProgram,
    _formula,
)


class _NeedChoice(Exception):
    def __init__(self, options):
        self.options = list(options)


def _value_from_domain(value):
    if value in {"True", "False"}:
        return value == "True"
    if value.isdigit():
        return int(value)
    return value


def _value_to_domain(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _execute_step_outcomes(src, vars_, state):
    outcomes = set()
    pending_paths = [()]
    original_choice = random.choice
    try:
        while pending_paths:
            path = pending_paths.pop()
            ns = {}
            exec(src, ns, ns)
            for i, (name, domain) in enumerate(vars_):
                ns[name] = _value_from_domain(domain[state[i]])

            choice_index = {"i": 0}

            def choice(options):
                i = choice_index["i"]
                choice_index["i"] += 1
                if i < len(path):
                    return path[i]
                raise _NeedChoice(options)

            ns["random"].choice = choice
            try:
                ns["step"]()
            except _NeedChoice as e:
                pending_paths.extend(path + (option,) for option in e.options)
                continue

            outcomes.add(
                tuple(domain.index(_value_to_domain(ns[name])) for name, domain in vars_)
            )
    finally:
        random.choice = original_choice
    return outcomes


def test_grammar_program_matches_derived_transitions_and_varies_structure():
    state = random.getstate()
    random.seed(0)
    try:
        cfg = CodeAnalysisConfig(n_vars=4, n_modes=4, domain_size=3, max_states=128)
        task = CodeAnalysis(config=cfg)
        constructs = set()
        type_signatures = set()
        shapes = set()
        graphs = set()

        def shape(node):
            return type(node).__name__, tuple(shape(child) for child in ast.iter_child_nodes(node))

        for _ in range(160):
            try:
                k = task._make_kripke()
            except _BadProgram:
                continue
            src = k.program
            types = {type(_value_from_domain(domain[0])).__name__ for _, domain in k.vars}
            type_signatures.add(tuple(sorted(types)))
            shapes.add(shape(ast.parse(src)))
            graphs.add(tuple(map(tuple, k.succ)))
            constructs |= {
                type(node).__name__
                for node in ast.walk(ast.parse(src))
                if isinstance(node, (ast.If, ast.Match, ast.Return, ast.Dict, ast.IfExp))
            }

            for sid in range(len(k.states)):
                expected = {k.states[j] for j in k.succ[sid]}
                got = _execute_step_outcomes(src, k.vars, k.states[sid])
                assert got == expected, (k.syntax, sid, expected, got)

        assert {"If", "Match", "Return", "Dict", "IfExp"} <= constructs
        assert len(type_signatures) >= 4
        assert len(shapes) >= 80
        assert len(graphs) >= 80
    finally:
        random.setstate(state)


def test_prompt_only_shows_state_table_and_choice_semantics_when_needed():
    task = CodeAnalysis()
    fields = dict(
        program="x = 0",
        state_variables="x",
        property_text="some execution can eventually reach x == 1",
        formula="EF(p0)",
        witness_kind="witness",
    )

    holds = task.render_prompt(SimpleNamespace(**fields, query_type="holds"))
    assert "Reachable states:" not in holds
    assert "Predicates:" not in holds
    assert "nondeterministic" not in holds
    assert "Formula:" in holds

    states = task.render_prompt(SimpleNamespace(**fields, query_type="states"))
    assert "Reachable states:" not in states
    assert "State tuples use (x)." in states

    fields["program"] = "x = random.choice([0, 1])"
    witness = task.render_prompt(SimpleNamespace(**fields, query_type="witness"))
    assert "Reachable states:" not in witness
    assert "nondeterministic transition" in witness


def test_nested_temporal_property_rendering_preserves_scope():
    task = CodeAnalysis()
    atom = _formula("atom", "p")
    af_ag = _formula("AF", _formula("AG", atom))
    ag_af = _formula("AG", _formula("AF", atom))

    assert task._render_property(af_ag) == (
        "on every execution, eventually (at every reachable state, (p))"
    )
    assert task._render_property(ag_af) == (
        "at every reachable state, (on every execution, eventually (p))"
    )
    assert task._render_property(af_ag) != task._render_property(ag_af)

    successors = [[1], [2], [0]]
    predicates = {"p": {2}}
    af_ag_sat, _ = task._sat(af_ag, successors, predicates)
    ag_af_sat, _ = task._sat(ag_af, successors, predicates)
    assert 0 not in af_ag_sat
    assert 0 in ag_af_sat


def test_state_answers_use_valuations_without_prompt_leakage():
    state = random.getstate()
    random.seed(4)
    try:
        task = CodeAnalysis()
        entries = [task.generate_entry() for _ in range(4)]
        for entry in entries:
            if entry.metadata.query_type not in {"states", "witness"}:
                continue
            value = ast.literal_eval(entry.answer)
            assert isinstance(value, list) and all(isinstance(item, tuple) for item in value)
            assert "Reachable states:" not in task.render_prompt(entry.metadata)
            assert task.score_answer(entry.answer, entry) == 1
    finally:
        random.setstate(state)


def test_holds_queries_require_nontrivial_temporal_reasoning(monkeypatch):
    state = random.getstate()
    random.seed(1)
    try:
        task = CodeAnalysis()
        original_shuffle = random.shuffle

        def holds_first(items):
            original_shuffle(items)
            if len(items) == 4 and all(isinstance(item, str) for item in items) and set(items) == {
                "holds", "states", "rank", "witness"
            }:
                items.remove("holds")
                items.insert(0, "holds")

        monkeypatch.setattr(random, "shuffle", holds_first)
        for _ in range(20):
            entry = task.generate_entry()
            metadata = entry.metadata
            assert metadata.query_type == "holds"
            assert metadata.root_operator in {"EX", "AX", "EF", "AF", "EG", "AG"}
            assert metadata.temporal_effort >= 2
            if (metadata.root_operator, entry.answer) in {("EX", "Yes"), ("AX", "No")}:
                assert metadata.mixed_initial_branches
        assert not hasattr(task, "_query_i")
    finally:
        random.setstate(state)
