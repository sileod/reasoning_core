from collections import Counter

from easydict import EasyDict as edict
from reasoning_core import get_task, list_tasks, score_answer
from reasoning_core.template import render_payload
import pytest

from reasoning_core.tasks.logic_depth import (
    Atom,
    DefeasibleNLI,
    DefeasibleNLIConfig,
    Denial,
    Not,
    PredSig,
    Rule,
    Theory,
    MultistepNLIConfig,
    _binary_query,
    chase,
    choose_naf_example,
    close_with,
    derivation_rules,
    generate_case,
    group_entity_facts,
    hard_target,
    naf_chase,
    opposite,
    render,
    render_naf,
    rule_text,
    stratify_rules,
    support_atoms,
    support_sources,
)


def test_multistep_nli_registers_and_generates():
    assert "multistep_nli" in list_tasks()
    assert "multistep_evidence_retrieval" in list_tasks()
    assert "multistep_abduction" in list_tasks()
    assert "logic_qa" in list_tasks()
    task = get_task("multistep_nli")
    seen = set()
    for _ in range(12):
        ex = task.generate_example(max_tokens=0)
        seen.add(ex.answer)
        assert ex.answer in {"Yes", "No", "Maybe"}
        assert ex.metadata.label in {"entailment", "contradiction", "neutral"}
        assert score_answer(ex.answer, ex) == 1
        assert task.score_answer(ex.answer + ".", ex) == 1
        assert task.score_answer("not a label", ex) == 0
    assert len(seen) >= 2
    assert sum(task._case_state["label_counts"].values()) == 12


def test_multistep_nli_retries_exhausted_search(monkeypatch):
    import reasoning_core.tasks.logic_depth as logic_depth

    real_generate_case = logic_depth.generate_case
    calls = 0

    def flaky_generate_case(*args, **kwargs):
        nonlocal calls
        calls += 1
        return (None, None) if calls < 3 else real_generate_case(*args, **kwargs)

    monkeypatch.setattr(logic_depth, "generate_case", flaky_generate_case)
    assert get_task("multistep_nli").generate_entry().answer in {"Yes", "No", "Maybe"}
    assert calls == 3


def test_multistep_nli_groups_entities_without_changing_facts():
    facts = [Atom("quiet", ("mary",)), Atom("quiet", ("paul",)), Atom("quiet", ("clara",), False)]
    theory = Theory(facts, [], [], {}, {"person": ("mary", "paul", "clara")})
    lines, _, _ = render(theory)
    assert group_entity_facts(theory, lines, 1) == ["mary and paul are quiet.", "clara is not quiet."]
    assert group_entity_facts(theory, lines, 0) == lines


def test_multistep_neutral_is_a_hard_near_miss_at_level_zero():
    cfg = MultistepNLIConfig()
    state = {}
    signs = Counter()
    for _ in range(12):
        case, _ = generate_case(cfg, ("neutral",), state)
        assert case is not None
        assert case.neutral_completion is not None
        signs[case.hyp.sign] += 1
        mentioned = {arg for atom in case.theory.facts for arg in atom.args}
        assert all(arg in mentioned for arg in case.hyp.args)
        completed = close_with(case.theory, [case.neutral_completion])
        target = case.hyp if case.hyp in completed.derivations else opposite(case.hyp)
        assert target in completed.derivations
        assert hard_target(target, completed.derivations, cfg)
        proof_size = len(support_atoms(target, completed.derivations)) + len(derivation_rules(target, completed.derivations))
        assert len(case.lines) == proof_size + cfg.n_distractors
    # The generator balances signs with slack: it only pushes back once one side leads by
    # more than six, and even then it lets one through a fifth of the time. Twelve draws
    # therefore come out even-ish, not even.
    assert signs[True] and signs[False]
    assert abs(signs[True] - signs[False]) <= 6


def test_defeasible_nli_registers_and_generates():
    assert "defeasible_nli" in list_tasks()
    task = get_task("defeasible_nli")
    ex = task.generate_example(max_tokens=0)
    assert ex.answer
    assert ex.metadata.naf_rule_count >= 1
    assert task.score_answer(ex.answer, ex) == 1
    verbose_prompt = (
        f"{render_payload({'premise': chr(10).join(ex.metadata.premise), 'hypothesis': ex.metadata.hypothesis})}\n"
        "Is the hypothesis true given the premise? The answer is Yes, No, or Maybe."
    )
    assert len(task.tokenizer.encode(ex.prompt)) < len(task.tokenizer.encode(verbose_prompt))
    nli = get_task("defeasible_nli")
    seen = {nli.generate_example(max_tokens=0).answer for _ in range(12)}
    assert seen <= {"Yes", "No", "Maybe"}
    assert len(seen) >= 2
    assert sum(nli._case_state["label_counts"].values()) == 12
    assert sum(nli._case_state["label_signs"].values()) == 12


def test_defeasible_nli_uses_compact_natural_language():
    sigs = {p: PredSig(p, ("person",)) for p in ("bird", "penguin", "ab_bird", "approved")}
    theory = Theory(
        facts=[Atom("bird", ("alice",)), Atom("penguin", ("bruno",))],
        rules=[
            Rule((Atom("penguin", ("?x",)),), Atom("ab_bird", ("?x",))),
            Rule(
                (Atom("bird", ("?x",)), Not(Atom("ab_bird", ("?x",)))),
                Atom("approved", ("?x",)),
            ),
        ],
        denials=[],
        pred_sigs=sigs,
        entities={"person": ("alice", "bruno")},
    )
    facts, rules = render_naf(theory)
    assert facts == "Alice is a bird.\nBruno is a penguin."
    assert rules == (
        "Penguins are abnormal.\n"
        "Birds are approved unless abnormal."
    )
    assert "ab bird" not in facts + rules

    task = DefeasibleNLI()
    prompt = task.render_prompt(edict(payload={"facts": facts, "rules": rules, "hypothesis": "Alice is approved."}))
    assert "An `unless` condition must be shown to block its rule" in prompt
    assert "it cannot be shown that" not in prompt


@pytest.mark.parametrize(
    ("sign_counts", "expected_sign"),
    [
        (Counter({("contradiction", True): 1}), False),
        (Counter({("contradiction", False): 1}), True),
    ],
)
def test_defeasible_contradiction_balances_both_hypothesis_signs(sign_counts, expected_sign):
    sigs = {p: PredSig(p, ("person",)) for p in ("known", "exception", "p", "q")}
    theory = Theory(
        facts=[Atom("known", ("alice",))],
        rules=[
            Rule(
                (Atom("known", ("?x",)), Not(Atom("exception", ("?x",)))),
                Atom("p", ("?x",)),
            ),
            Rule(
                (Atom("known", ("?x",)), Not(Atom("exception", ("?x",)))),
                Atom("q", ("?x",), False),
            ),
        ],
        denials=[],
        pred_sigs=sigs,
        entities={"person": ("alice",)},
    )
    res = naf_chase(theory)
    cfg = DefeasibleNLIConfig(min_target_depth=1, max_target_depth=1, min_naf_rules_in_proof=1)
    choice = choose_naf_example(
        theory,
        res,
        cfg,
        Counter({"entailment": 1, "neutral": 1}),
        sign_counts,
    )
    assert choice[0] == "contradiction"
    assert choice[1].sign is expected_sign


def test_defeasible_nli_level2_is_not_all_maybe():
    cfg = DefeasibleNLIConfig()
    cfg.set_level(2)
    task = DefeasibleNLI(config=cfg)
    seen = {task.generate().answer for _ in range(40)}
    assert seen <= {"Yes", "No", "Maybe"}
    assert seen != {"Maybe"}


def test_defeasible_nli_level2_balanced_batch_completes():
    task = get_task("defeasible_nli")
    batch = task.generate_balanced_batch(batch_size=12, level=2, max_tokens=0)
    assert len(batch) == 12
    assert {ex.answer for ex in batch} == {"Yes", "No", "Maybe"}


def test_multistep_evidence_retrieval_generates():
    task = get_task("multistep_evidence_retrieval")
    ex = task.generate_example(max_tokens=0)
    assert ex.answer
    assert all(x.isdigit() for x in ex.answer.split())
    assert ex.metadata.label in {"entailment", "contradiction"}
    assert ex.metadata.necessary_indices
    assert len(ex.metadata.valid_supports) == 1
    assert ex.metadata.support_indices == ex.metadata.valid_supports[0]
    assert ex.metadata.support_indices == ex.metadata.necessary_indices
    assert task.score_answer(ex.answer, ex) == 1
    assert task.score_answer("[]", ex) == float(ex.answer == "[]")


def test_multistep_abduction_generates():
    task = get_task("multistep_abduction")
    ex = task.generate_example(max_tokens=0)
    assert ex.answer
    assert all(x.isdigit() for x in ex.answer.split())
    assert ex.metadata.label in {"entailment", "contradiction"}
    assert ex.metadata.candidates
    assert task.score_answer(ex.answer, ex) == 1


def test_logic_qa_generates():
    task = get_task("logic_qa")
    seen = set()
    for _ in range(8):
        ex = task.generate_example(max_tokens=0)
        seen.add(ex.metadata.answer_mode)
        assert ex.metadata.answer_mode in {"count", "list"}
        assert ex.metadata.question
        if ex.answer not in {"0", "none"}:
            assert ex.metadata.support_indices
        assert task.score_answer(ex.answer, ex) == 1
    assert seen


def test_logic_qa_default_queries_have_a_multistep_nonempty_answer():
    task = get_task("logic_qa")
    for _ in range(8):
        ex = task.generate_example(max_tokens=0)
        assert ex.answer not in {"0", "none"}
        assert ex.metadata.hard_answer_depths
        assert min(ex.metadata.hard_answer_depths) >= 2


def test_binary_logic_qa_questions_are_proof_oriented_and_grammatical():
    assert _binary_query("aunt_or_uncle", "alice", False, "kinship") == (
        "Which other entities can alice be shown to be an aunt or uncle of?"
    )
    assert _binary_query("contains", "box", True, "spatial") == (
        "How many other entities can box be shown to contain?"
    )
    assert _binary_query("helps", "clara", False, "surface") == (
        "Which other entities can clara be shown to stand in the helps relation to?"
    )


def test_chase_keeps_minimal_depth_derivation():
    sigs = {p: PredSig(p, ("person",)) for p in ("a", "b", "c")}
    theory = Theory(
        facts=[Atom("a", ("alice",))],
        rules=[
            Rule((Atom("a", ("?x",)),), Atom("b", ("?x",)), shape="u_imp"),
            Rule((Atom("b", ("?x",)),), Atom("c", ("?x",)), shape="u_imp"),
            Rule((Atom("a", ("?x",)),), Atom("c", ("?x",)), shape="u_imp"),
        ],
        denials=[],
        pred_sigs=sigs,
        entities={"person": ("alice",)},
    )
    res = chase(theory, max_depth=4)
    assert not res.inconsistent
    assert res.derivations[Atom("c", ("alice",))].depth == 1


def test_existing_chase_semantics_unchanged_for_classical_negation():
    sigs = {p: PredSig(p, ("person",)) for p in ("a", "b")}
    theory = Theory(
        facts=[Atom("a", ("alice",)), Atom("a", ("bruno",), False)],
        rules=[Rule((Atom("a", ("?x",)),), Atom("b", ("?x",)), shape="u_imp")],
        denials=[],
        pred_sigs=sigs,
        entities={"person": ("alice", "bruno")},
    )
    res = chase(theory, max_depth=None)
    assert res.closure == {
        Atom("a", ("alice",)),
        Atom("a", ("bruno",), False),
        Atom("b", ("alice",)),
    }


def test_naf_rejects_unbound_negative_variables():
    sigs = {p: PredSig(p, ("person",)) for p in ("flagged", "trusted")}
    theory = Theory(
        facts=[Atom("flagged", ("alice",), False)],
        rules=[Rule((Not(Atom("flagged", ("?x",))),), Atom("trusted", ("?x",)))],
        denials=[],
        pred_sigs=sigs,
        entities={"person": ("alice",)},
    )
    with pytest.raises(ValueError, match="unsafe NAF rule"):
        naf_chase(theory, max_depth=None)


def test_naf_distinct_from_classical_negative_atom_when_safely_bound():
    sigs = {p: PredSig(p, ("person",)) for p in ("person", "flagged", "trusted")}
    theory = Theory(
        facts=[Atom("person", ("alice",)), Atom("flagged", ("alice",), False)],
        rules=[Rule((Atom("person", ("?x",)), Not(Atom("flagged", ("?x",)))), Atom("trusted", ("?x",)))],
        denials=[],
        pred_sigs=sigs,
        entities={"person": ("alice",)},
    )
    res = naf_chase(theory, max_depth=None)
    assert Atom("trusted", ("alice",)) in res.closure


def test_exception_blocks_default_to_maybe():
    sigs = {p: PredSig(p, ("person",)) for p in ("trained", "flagged", "trusted")}
    theory = Theory(
        facts=[Atom("trained", ("alice",)), Atom("flagged", ("alice",))],
        rules=[Rule((Atom("trained", ("?x",)), Not(Atom("flagged", ("?x",)))), Atom("trusted", ("?x",)))],
        denials=[],
        pred_sigs=sigs,
        entities={"person": ("alice",)},
    )
    res = naf_chase(theory, max_depth=None)
    assert Atom("trusted", ("alice",)) not in res.closure
    assert Atom("trusted", ("alice",), False) not in res.closure


def test_explicit_negative_gives_no_under_naf():
    sigs = {p: PredSig(p, ("person",)) for p in ("trained", "flagged", "trusted")}
    theory = Theory(
        facts=[Atom("trained", ("alice",)), Atom("flagged", ("alice",))],
        rules=[
            Rule((Atom("trained", ("?x",)), Not(Atom("flagged", ("?x",)))), Atom("trusted", ("?x",))),
            Rule((Atom("flagged", ("?x",)),), Atom("trusted", ("?x",), False)),
        ],
        denials=[],
        pred_sigs=sigs,
        entities={"person": ("alice",)},
    )
    res = naf_chase(theory, max_depth=None)
    assert Atom("trusted", ("alice",)) not in res.closure
    assert Atom("trusted", ("alice",), False) in res.closure


def test_naf_stratification_rejects_negative_cycle():
    rules = [
        Rule((Atom("dom", ("?x",)), Not(Atom("q", ("?x",)))), Atom("p", ("?x",))),
        Rule((Atom("dom", ("?x",)), Not(Atom("p", ("?x",)))), Atom("q", ("?x",))),
    ]
    assert stratify_rules(rules) is None


def test_naf_stratification_allows_positive_recursion():
    rules = [
        Rule((Atom("parent", ("?x", "?y")),), Atom("ancestor", ("?x", "?y"))),
        Rule((Atom("parent", ("?x", "?y")), Atom("ancestor", ("?y", "?z"))), Atom("ancestor", ("?x", "?z"))),
    ]
    assert stratify_rules(rules) is not None


def test_naf_lower_stratum_exception_allowed_and_blocks():
    sigs = {p: PredSig(p, ("person",)) for p in ("bird", "penguin", "ab_bird", "flies")}
    theory = Theory(
        facts=[Atom("penguin", ("tweety",)), Atom("bird", ("tweety",))],
        rules=[
            Rule((Atom("bird", ("?x",)), Not(Atom("ab_bird", ("?x",)))), Atom("flies", ("?x",))),
            Rule((Atom("penguin", ("?x",)),), Atom("ab_bird", ("?x",))),
        ],
        denials=[],
        pred_sigs=sigs,
        entities={"person": ("tweety",)},
    )
    assert stratify_rules(theory.rules) is not None
    res = naf_chase(theory, max_depth=None)
    assert Atom("ab_bird", ("tweety",)) in res.closure
    assert Atom("flies", ("tweety",)) not in res.closure


def test_chase_saturates_when_depth_is_none():
    sigs = {p: PredSig(p, ("person",)) for p in ("a", "b", "c", "d")}
    rules = [
        Rule((Atom("a", ("?x",)),), Atom("b", ("?x",)), shape="u_imp"),
        Rule((Atom("b", ("?x",)),), Atom("c", ("?x",)), shape="u_imp"),
        Rule((Atom("c", ("?x",)),), Atom("d", ("?x",)), shape="u_imp"),
    ]
    theory = Theory([Atom("a", ("alice",))], rules, [], sigs, {"person": ("alice",)})
    assert Atom("d", ("alice",)) not in chase(theory, max_depth=2).closure
    assert Atom("d", ("alice",)) in chase(theory, max_depth=None).closure
    assert Atom("d", ("alice",)) in close_with(theory, []).closure


def test_support_sources_include_facts_and_rules():
    sigs = {p: PredSig(p, ("person",)) for p in ("a", "b")}
    rule = Rule((Atom("a", ("?x",)),), Atom("b", ("?x",)), shape="u_imp")
    theory = Theory([Atom("a", ("alice",))], [rule], [], sigs, {"person": ("alice",)})
    res = chase(theory, max_depth=None)
    _, source, _ = render(theory)
    assert support_sources(Atom("b", ("alice",)), res.derivations, source) == {0, 1}


def test_negative_composition_head_uses_generic_signed_rule_text():
    rule = Rule(
        (Atom("helps", ("?x", "?y")), Atom("trusts", ("?y", "?z"))),
        Atom("advises", ("?x", "?z"), False),
        shape="composition",
    )
    text, _ = rule_text(rule, rid=0)
    assert "x does not stand in the advises relation to z" in text
    assert "the first advises the third" not in text


def test_negative_converse_head_uses_generic_signed_rule_text():
    rule = Rule(
        (Atom("helps", ("?x", "?y")),),
        Atom("trusts", ("?y", "?x"), False),
        shape="converse",
    )
    text, _ = rule_text(rule, rid=0)
    assert "y does not stand in the trusts relation to x" in text
    assert "the second trusts the first" not in text


def test_naf_rule_text_distinguishes_not_derivable_from_classical_not():
    rule = Rule(
        (Atom("trained", ("?x",)), Not(Atom("trusted", ("?x",), False))),
        Atom("approved", ("?x",)),
    )
    text, _ = rule_text(rule, rid=0)
    assert "it cannot be shown that x is not trusted" in text
    assert "not trusted" in text


def test_spatial_left_of_cycle_is_inconsistent_via_derived_self_loop():
    sigs = {"left_of": PredSig("left_of", ("item", "item"))}
    theory = Theory(
        facts=[Atom("left_of", ("a", "b")), Atom("left_of", ("b", "a"))],
        rules=[
            Rule(
                (Atom("left_of", ("?x", "?y")), Atom("left_of", ("?y", "?z"))),
                Atom("left_of", ("?x", "?z")),
                shape="composition",
            )
        ],
        denials=[Denial((Atom("left_of", ("?x", "?x")),))],
        pred_sigs=sigs,
        entities={"item": ("a", "b")},
        domain_pack="spatial",
    )
    res = chase(theory, max_depth=None)
    assert res.inconsistent
    assert Atom("left_of", ("a", "a")) in res.closure

def test_spatial_acyclic_left_of_chain_remains_consistent():
    sigs = {"left_of": PredSig("left_of", ("item", "item"))}
    theory = Theory(
        facts=[Atom("left_of", ("a", "b")), Atom("left_of", ("b", "c"))],
        rules=[
            Rule(
                (Atom("left_of", ("?x", "?y")), Atom("left_of", ("?y", "?z"))),
                Atom("left_of", ("?x", "?z")),
                shape="composition",
            )
        ],
        denials=[Denial((Atom("left_of", ("?x", "?x")),))],
        pred_sigs=sigs,
        entities={"item": ("a", "b", "c")},
        domain_pack="spatial",
    )
    res = chase(theory, max_depth=None)
    assert not res.inconsistent
    assert Atom("left_of", ("a", "c")) in res.closure
