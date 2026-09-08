from hypothesis import given, settings, strategies as st

from reasoning_core.template import Entry
from reasoning_core.tasks.math_scratch import (
    MathScratchCoreSelect,
    MathScratchEntailment,
    MathScratchNormalize,
    ScratchConfig,
    ScratchRule,
    _certify_equivalent,
    _certify_inequivalent,
    _match,
    _normalize,
    _ordered,
    _rewrite_once,
    _sample_negative_rhs,
    _sample_ground_term,
    _sample_positive_instance,
    _sample_world,
    _size,
    _subst,
    _valid_rule,
    _vars,
)


def _from_data(data):
    if isinstance(data, list):
        return tuple([data[0], *[_from_data(x) for x in data[1:]]])
    return data


def _rules_from_metadata(metadata):
    return [
        ScratchRule(
            rule["name"],
            _from_data(rule["lhs"]),
            _from_data(rule["rhs"]),
            rule.get("mode", "size_decreasing"),
            tuple(rule.get("precedence", ())),
        )
        for rule in metadata.raw_rules
    ]


def test_nonlinear_match_requires_equal_subterms():
    pattern = ("g0", "?X", "?X")

    assert _match(pattern, ("g0", "c1", "c1")) == {"?X": "c1"}
    assert _match(pattern, ("g0", "c1", "c2")) is None


def test_rule_validator_rejects_variable_duplication_growth():
    lhs = ("f0", ("g0", "?X", "?Y"))
    rhs = ("g0", "?X", "?X")

    assert not _valid_rule(lhs, rhs)


def test_canonical_normalize_uses_outermost_position_before_rule_order():
    rules = [
        ScratchRule("r1", ("f0", "?X"), "?X"),
        ScratchRule("r2", ("g0", "?X", "c0"), "?X"),
    ]
    term = ("g0", ("f0", "c1"), "c0")

    nf, trace = _normalize(term, rules)

    assert nf == "c1"
    assert [step[1] for step in trace] == ["r2", "r1"]


def test_negative_displayed_rhs_is_normal_form():
    config = ScratchConfig()
    rules = _sample_world(config)
    inst = _sample_positive_instance(rules, config)
    rhs_bad, _ = _sample_negative_rhs(inst.lhs, inst.rhs, rules, config)

    assert rhs_bad != inst.rhs
    assert _rewrite_once(rhs_bad, rules) is None


def test_coreselect_certifies_single_sufficient_option():
    task = MathScratchCoreSelect()
    entry = task.generate_example()
    rules = _rules_from_metadata(entry.metadata)
    sufficient = []

    for i, option in enumerate(entry.metadata.options):
        ok = _certify_equivalent(
            _from_data(entry.metadata.raw_left),
            _from_data(entry.metadata.raw_right),
            _ordered(option, rules),
            task.config,
        )
        if ok:
            sufficient.append(i)

    assert sufficient == [ord(entry.answer) - ord("A")]
    assert entry.metadata.options[sufficient[0]] == entry.metadata.core
    for i, option in enumerate(entry.metadata.options):
        if i != sufficient[0]:
            assert _certify_inequivalent(
                _from_data(entry.metadata.raw_left),
                _from_data(entry.metadata.raw_right),
                _ordered(option, rules),
                task.config,
            )


def test_score_extractors_use_last_standalone_label():
    core = MathScratchCoreSelect()
    entail = MathScratchEntailment()

    assert core.score_answer("A. no\nAnswer: B", Entry({}, "B")) == 1.0
    assert core.score_answer("ANSWER: B", Entry({}, "B")) == 1.0
    assert entail.score_answer("It is true. Final answer: False", Entry({}, "False")) == 1.0


def test_prompts_request_anchored_answer():
    assert "Answer:" in MathScratchEntailment().generate_example().prompt
    assert "Answer:" in MathScratchCoreSelect().generate_example().prompt
    assert "Answer:" in MathScratchNormalize().generate_example().prompt


ground_terms = st.recursive(
    st.sampled_from(["c0", "c1", "c2"]),
    lambda children: st.one_of(
        st.tuples(st.just("f0"), children),
        st.tuples(st.just("g0"), children, children),
    ),
    max_leaves=12,
)


@given(ground_terms)
@settings(max_examples=30)
def test_normalize_is_idempotent_on_strict_rules(term):
    rules = [
        ScratchRule("r1", ("f0", "?X"), "?X"),
        ScratchRule("r2", ("g0", "?X", "c0"), "?X"),
        ScratchRule("r3", ("g0", "?X", "?X"), "?X"),
    ]

    nf, _ = _normalize(term, rules)
    nf2, _ = _normalize(nf, rules)

    assert nf2 == nf


def test_match_then_subst_reconstructs_matched_term():
    pattern = ("g0", ("f0", "?X"), "?Y")
    term = ("g0", ("f0", "c1"), ("g0", "c2", "c2"))

    env = _match(pattern, term)

    assert env is not None
    assert _subst(pattern, env) == term


def test_accepted_rules_strictly_decrease_random_ground_instances():
    config = ScratchConfig()
    rules = _sample_world(config)

    for rule in rules:
        for _ in range(5):
            env = {var: _sample_ground_term(config, 2) for var in _vars(rule.lhs)}
            assert _size(_subst(rule.rhs, env)) < _size(_subst(rule.lhs, env))


def test_lpo_rule_can_increase_size_and_duplicate_variables():
    config = ScratchConfig(rule_mode="lpo")
    precedence = ("g0", "g1", "c0", "c1")
    lhs = ("g0", "?X", ("g1", "?Y", "?Z"))
    rhs = ("g1", ("g0", "?X", "?Y"), ("g0", "?X", "?Z"))
    rule = ScratchRule("r1", lhs, rhs, "lpo", precedence)

    assert _valid_rule(lhs, rhs, config, precedence)
    nf, trace = _normalize(("g0", "c0", ("g1", "c1", "c1")), [rule])
    assert nf == ("g1", ("g0", "c0", "c1"), ("g0", "c0", "c1"))
    assert _size(nf) > _size(("g0", "c0", ("g1", "c1", "c1")))
    assert trace[0][2] == ()


def test_used_rules_replay_positive_trace():
    config = ScratchConfig()
    rules = _sample_world(config)
    inst = _sample_positive_instance(rules, config)

    full_nf, full_trace = _normalize(inst.lhs, rules, config.max_norm_steps)
    used_nf, used_trace = _normalize(inst.lhs, _ordered(inst.used, rules), config.max_norm_steps)

    assert full_nf == used_nf == inst.rhs
    assert [(r, p, a) for _, r, p, a in used_trace] == [(r, p, a) for _, r, p, a in full_trace]


def test_guard_passing_world_agrees_with_random_strategies():
    config = ScratchConfig()
    rules = _sample_world(config)

    for _ in range(8):
        term = _sample_ground_term(config)
        canonical, _ = _normalize(term, rules, config.max_norm_steps)
        random_nf, _ = _normalize(term, rules, config.max_norm_steps, random_strategy=True)
        assert random_nf == canonical


def test_normalize_task_scores_parsed_answer():
    task = MathScratchNormalize()
    entry = task.generate_example()

    assert task.score_answer(f"Answer: {entry.answer}", entry) == 1.0
    assert task.score_answer("Answer: definitely_not_the_nf", entry) == 0.0


def test_lpo_mode_generates_examples():
    config = ScratchConfig(rule_mode="lpo", max_norm_steps=96, max_intermediate_size=1024)
    ex = MathScratchNormalize(config).generate_example()

    assert ex.answer == ex.metadata.normal_form
    assert ex.metadata.rule_mode == "lpo"
