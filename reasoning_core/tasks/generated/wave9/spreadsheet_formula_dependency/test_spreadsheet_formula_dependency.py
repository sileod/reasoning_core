import pytest

from reasoning_core.tasks.generated.wave9.spreadsheet_formula_dependency.spreadsheet_formula_dependency import (
    SpreadsheetFormulaDependency,
    _eval_arith,
    _solve_formula,
    _refs_of,
    _cell_names,
)


def test_arith_basic():
    assert _eval_arith("1+2+3") == 6
    assert _eval_arith("2*3+4") == 10
    assert _eval_arith("5+2*3") == 11
    assert _eval_arith("7") == 7


def test_cell_names():
    assert _cell_names(3) == ["A1", "A2", "A3"]


def test_refs_of_range():
    assert _refs_of("sum(A1:A3)") == {"A1", "A2", "A3"}


def test_solve_formula_range():
    vals = {"A1": 5, "A2": 3, "A3": 2}
    assert _solve_formula("sum(A1:A3)", vals) == 10
    assert _solve_formula("max(A1:A3)", vals) == 5


def test_end_to_end():
    task = SpreadsheetFormulaDependency()
    for _ in range(20):
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1
        assert ex.answer.isdigit()


def test_wrong_answer_scores_zero():
    task = SpreadsheetFormulaDependency()
    ex = task.generate_example()
    assert task.score_answer(str(int(ex.answer) + 1), ex) == 0
    assert task.score_answer("", ex) == 0
