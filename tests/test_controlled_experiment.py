from types import SimpleNamespace

from reasoning_core.evaluation.training.controlled_experiment import control_spec, row_filter


def test_row_filter_top_level_fields():
    row = {"task": "bayesian_intervention", "mode": "instruct", "level": 3}
    assert row_filter(row, task="bayesian_intervention", mode="instruct", level="3")
    assert not row_filter(row, task="logic_nli")


def test_row_filter_metadata_fields():
    row = {"metadata": '{"_task":"logic_nli","mode":"cot","_level":2}'}
    assert row_filter(row, task="logic_nli", mode="cot", level="2")
    assert not row_filter(row, level="3")


def test_control_spec_is_task_only():
    assert control_spec(SimpleNamespace(aux_task="logic_nli", aux_mode="", aux_level="")) == {
        "aux_task": "logic_nli",
        "aux_mode": None,
        "aux_level": None,
    }
    assert control_spec(SimpleNamespace(aux_task="", aux_mode="cot", aux_level="3")) == {
        "aux_task": None,
        "aux_mode": "cot",
        "aux_level": "3",
    }
