from reasoning_core.downstream_eval.logic import evaluate_logic


def test_logic_smoke():
    scores = evaluate_logic(
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        tasks=("logiqa2_nli", "rte"),
        limit=2,
        batch_size=1,
        device="cpu",
        timeout_s=180,
    )
    assert scores
