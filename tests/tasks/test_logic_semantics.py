from reasoning_core.tasks.logic_semantics import get_cot


def test_get_cot_handles_named_inputs_without_numeric_ids():
    proof = "\n".join([
        "1. ! [X0] : anywhere(X0) [input anywhere_ax]",
        "2. p(a) [input p0]",
        "3. q(a) [input 1]",
        "4. ~p(a) [input hyp]",
        "5. r(a) [input background2]",
    ])

    assert get_cot(proof).splitlines() == [
        "1. [H] ! [X0] : anywhere(X0)",
        "2. [P0] p(a)",
        "3. [P1] q(a)",
        "4. [H] ~p(a)",
        "5. [H] r(a)",
    ]
