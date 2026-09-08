import pytest

from reasoning_core.template import Problem, edict
from reasoning_core.tasks.math_lean import LeanConfig
from reasoning_core.tasks import math_lean

if not hasattr(math_lean, "LeanProofRepair"):
    pytest.skip("LeanProofRepair is no longer an active task", allow_module_level=True)

from reasoning_core.tasks.math_lean import LeanProofRepair


def test_lean_proof_repair_scores_exact_candidate_body():
    task = LeanProofRepair(LeanConfig(use_mathlib=True))
    entry = Problem(
        edict(
            replacements=[
                "intro x hx; exact ⟨hx.1, h0 hx.2⟩",
                "rfl",
            ],
            use_mathlib=True,
        ),
        "intro x hx; exact ⟨hx.1, h0 hx.2⟩",
    )
    pred = "intro x ⟨hxt, hxu⟩\nexact ⟨hxt, h0 hxu⟩"
    assert task.score_answer(entry.answer, entry) == 1.0
    assert task.score_answer("1. intro x hx; exact ⟨hx.1, h0 hx.2⟩", entry) == 1.0
    assert task.score_answer("```lean\nintro x hx; exact ⟨hx.1, h0 hx.2⟩\n```", entry) == 1.0
    assert task.score_answer(pred, entry) == 0.0


def test_lean_proof_repair_prompt_shows_candidate_replacements():
    task = LeanProofRepair(LeanConfig(use_mathlib=True))
    prompt = task.prompt(
        edict(
            broken="theorem ex : True := by\n  rfl\n",
            replacements=["exact True.intro", "rfl"],
            use_mathlib=True,
        )
    )
    assert "CANDIDATE REPLACEMENTS" in prompt
    assert "The answer is exactly one candidate body" in prompt
