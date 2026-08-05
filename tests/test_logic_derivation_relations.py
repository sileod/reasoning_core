from reasoning_core.tasks.logic_depth import Atom, Rule, Theory, chase
from reasoning_core.tasks.logic_derivation import canonical_trace


def relation_trace(sign=True):
    fact = Atom("p0", ("Alice",))
    rule = Rule(
        (Atom("p0", ("?x",)),),
        Atom("r0", ("?x", "Bob"), sign),
    )
    theory = Theory(
        facts=[fact],
        rules=[rule],
        denials=[],
        pred_sigs={},
        entities={},
        domain_pack="abstract",
    )
    res = chase(theory)
    target = next(atom for atom in res.closure if atom not in theory.facts)
    trace = canonical_trace(theory, res, {fact: 0, rule: 1}, target)
    assert trace.unique
    return trace.answer


def test_abstract_binary_conclusion_uses_prompt_surface_text():
    assert relation_trace() == "1: 0 => Alice is alpha-linked to Bob"


def test_negative_abstract_binary_conclusion_uses_prompt_surface_text():
    assert relation_trace(False) == "1: 0 => Alice is not alpha-linked to Bob"
