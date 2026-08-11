from reasoning_core import list_tasks
from scripts.build_gallery import sort_task_names


def test_gallery_keeps_math_and_logic_tasks_in_their_clusters():
    ordered = sort_task_names(list_tasks())
    positions = {name: ordered.index(name) for name in (
        "equation_system",
        "combinatorics_formula",
        "function_manipulation",
        "lean_missing_line",
        "logic_qa",
        "logic_derivation",
        "planning",
    )}

    assert positions["equation_system"] < positions["combinatorics_formula"]
    assert positions["combinatorics_formula"] < positions["function_manipulation"]
    assert positions["function_manipulation"] < positions["lean_missing_line"]
    assert positions["logic_qa"] < positions["logic_derivation"]
    assert positions["logic_derivation"] < positions["planning"]


def test_combinatorics_task_uses_synthesis_name_only():
    tasks = list_tasks()

    assert "combinatorics_formula" in tasks
    assert "combinatorics_formula_selection" not in tasks
