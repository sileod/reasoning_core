from reasoning_core.tasks.deprecated import math_rocq as mr


def test_candidate_labels_batched_parses_compiler_markers(monkeypatch):
    def fake_check_rocq(source, timeout):
        return True, "RC_CAND_1_TRUE\nRC_CAND_2_FALSE\n", ""

    monkeypatch.setattr(mr, "check_rocq", fake_check_rocq)

    labels = mr._candidate_labels_batched(
        "From Stdlib Require Import ZArith Ring.\n"
        "Open Scope Z_scope.\n\n"
        "Theorem target (a b : Z) : a + b = b + a.",
        ["ring.", "lia."],
        timeout=20,
    )

    assert labels == [True, False]


def test_proof_repair_distractors_match_header_imports():
    config = mr.RocqConfig()

    ring = mr._proof_ring(config)
    tauto = mr._proof_tauto(config)

    assert "Lia" not in ring.header
    assert all("lia" not in cand for cand in ring.distractors)
    assert "Lia" not in tauto.header
    assert all("lia" not in cand for cand in tauto.distractors)
