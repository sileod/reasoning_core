import inspect
import textwrap

from reasoning_core.template import Entry, edict
from reasoning_core.tasks.code_program_synthesis import ProgramSynthesis


def _entry(solution, examples, *, holdout=(), prompt_ops=("concat",)):
    return Entry(
        metadata=edict(
            solution_expr=solution,
            solution_function=f"def f(s: str) -> str:\n    return {solution}",
            io_pairs=examples,
            holdout=holdout,
            prompt_ops=prompt_ops,
        ),
        answer=f"def f(s: str) -> str:\n    return {solution}",
    )


def test_program_scoring_reserves_one_for_the_minimum_cost_ast():
    task = ProgramSynthesis()
    entry = _entry('(s + "_")', [("a", "a_")], holdout=[("b", "b_")])

    assert task.score_answer("def f(s: str) -> str:\n    return s + '_'", entry) == 1.0
    score = task.score_answer('return ((s + "") + "_")', entry)
    assert score == 0.9


def test_program_scoring_gives_behavioral_partial_credit_to_valid_dsl():
    task = ProgramSynthesis()
    entry = _entry(
        '(s + "-").replace("-", "_", 1)',
        [(" ", " _"), ("abc", "abc_"), ("-", "_-")],
        prompt_ops=("concat", "replace1"),
    )

    score = task.score_answer("def f(s: str) -> str:\n    return s + '_'", entry)
    assert score == 0.6
    assert task.score_answer("return s + '--'", entry) == 0.0


def test_program_scoring_caps_safe_out_of_dsl_shorthand_below_valid_dsl():
    task = ProgramSynthesis()
    entry = _entry(
        's.replace("_", ("-" + "-"), 1)',
        [("aa", "aa"), ("_", "--")],
        holdout=[("a_b", "a--b")],
        prompt_ops=("concat", "replace1"),
    )

    score = task.score_answer("return s.replace('_', '--', 1)", entry)
    assert score == 0.45


def test_program_scorer_works_when_source_extracted_standalone():
    namespace = {}
    exec(textwrap.dedent(inspect.getsource(ProgramSynthesis.score_answer)), namespace)
    entry = _entry('(s + "_")', [("a", "a_")])

    assert namespace["score_answer"](object(), entry.answer, entry) == 1.0
