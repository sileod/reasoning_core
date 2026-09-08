from reasoning_core.tasks.binding import RewriteSystem, RewriteSystemConfig


def test_rewrite_trace_is_metadata_cot_not_answer():
    config = RewriteSystemConfig()
    problem = RewriteSystem(config).generate_example()

    assert problem.answer == problem.metadata.normal_form
    assert problem.metadata.cot.startswith("-> ")
    assert problem.metadata.cot.endswith(f"normal_form: {problem.answer}")
    assert "-> " not in problem.answer
    assert "rewrite trace" not in problem.prompt
    assert RewriteSystem(config).score_answer(problem.answer, problem) == 1.0
