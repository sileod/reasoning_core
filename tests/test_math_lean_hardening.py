from reasoning_core.tasks.math_lean import (
    LeanCandidateCompilation,
    LeanConfig,
    LeanMissingLine,
    get_runner,
)


def test_missing_line_uses_exact_line_and_unique_checked_near_misses():
    config = LeanConfig(use_mathlib=False)
    config.set_level(2)
    task = LeanMissingLine(config)
    entry = task.generate_example(max_tokens=0)

    assert entry.answer == entry.metadata.correct_line
    assert len(entry.metadata.available_lines) == min(8, config.n_candidates)
    compiling = [
        i for i, line in enumerate(entry.metadata.available_lines, 1)
        if get_runner(False).check(entry.metadata.template.replace("__ANSWER__", line))[0]
    ]
    assert compiling == [entry.metadata.correct_index]
    assert entry.metadata.candidate_compiles == [
        i in compiling for i in range(1, len(entry.metadata.available_lines) + 1)
    ]


def test_candidate_compilation_uses_a_checked_proof_corruption_pair():
    task = LeanCandidateCompilation(LeanConfig(use_mathlib=False))
    entry = task.generate_example(max_tokens=0)
    candidate_code = entry.metadata.theorem.replace(
        "  ?\n", f"  {entry.metadata.candidate}\n"
    )
    paired_code = entry.metadata.theorem.replace(
        "  ?\n", f"  {entry.metadata.paired_candidate}\n"
    )

    assert get_runner(False).check(candidate_code)[0] == (entry.answer == "True")
    assert get_runner(False).check(paired_code)[0] == (entry.answer == "False")
    assert entry.metadata.candidate_similarity >= 0.5
