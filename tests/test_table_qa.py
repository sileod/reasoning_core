from collections import Counter
from datetime import date

import numpy as np
import pandas as pd

import reasoning_core.tasks.table_qa as table_qa
from reasoning_core.tasks.table_qa import (
    TableEquivalence, TableQA, TableQAConfig, canonical_scalar, canonical_table,
    corrupt_table, equivalence_display, render_nulls,
)


def test_canonical_scalar_conventions():
    assert canonical_scalar(date(2026, 7, 12)) == "2026-07-12"
    assert canonical_scalar(np.bool_(True)) == "true"
    assert canonical_scalar(False) == "false"
    assert canonical_scalar(np.nan) == "NULL"
    assert canonical_scalar(1234.5) == "1234.5"


def scalar_prompt(kind):
    return TableQA().render_prompt({
        "is_scalar": True,
        "scalar_kind": kind,
        "tables": ["x"],
        "query": "SELECT 1",
    })


def test_table_qa_states_only_the_relevant_scalar_convention():
    assert "YYYY-MM-DD" in scalar_prompt("date")
    assert "`true` or `false`" in scalar_prompt("bool")
    assert "YYYY-MM-DD" not in scalar_prompt("bool")
    assert "literal NULL" in scalar_prompt("null")
    assert "without display formatting" in scalar_prompt("number")
    assert "The answer is the result as a single value." in scalar_prompt(None)


def test_table_qa_declares_and_uses_unambiguous_null_marker():
    rendered, metadata = render_nulls(pd.Series([None, "-", "NA", "null", ""]))

    assert rendered.tolist() == ["—", "-", "NA", "null", ""]
    assert metadata == {"null": "—"}
    assert "In this table, — represents SQL NULL." in scalar_prompt(None)


def test_table_equivalence_is_stateless_and_batch_balanced():
    task = TableEquivalence()
    batch = task.generate_balanced_batch(batch_size=4)

    assert not hasattr(task, "_same_next")
    assert Counter(problem.answer for problem in batch) == {"Yes": 2, "No": 2}


def test_equivalence_normalization_is_controlled_and_duplicate_sensitive():
    df = pd.DataFrame({"number": [1000, None], "date": [date(2026, 7, 13)] * 2})

    assert equivalence_display(df, "plain").iloc[0].tolist() == ["1000.0", "2026-07-13"]
    assert equivalence_display(df, "formatted").iloc[0].tolist() == ["1,000.00", "Jul 13, 2026"]
    assert equivalence_display(df, "plain").iloc[1, 0] == "—"
    assert equivalence_display(df, "formatted").iloc[1, 0] == "NULL"
    assert canonical_table(df) != canonical_table(pd.concat([df, df.iloc[[0]]], ignore_index=True))


def test_multiple_corruptions_are_certified_inequivalent():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    corrupted, mutations = corrupt_table(df, count=3)

    assert len(mutations) == 3
    assert canonical_table(corrupted) != canonical_table(df)


def test_table_qa_conditions_data_on_each_query_family(monkeypatch):
    task = TableQA(TableQAConfig(num_rows=8, num_columns=6, complexity=3))

    for family in ("count", "arithmetic", "grouped_arithmetic"):
        monkeypatch.setattr(table_qa, "_sample_query_family", lambda _config: family)
        entry = task.generate_entry()
        spec = entry.metadata["query_spec"]
        checks = spec["feature_checks"]

        assert spec["query_conditioned"]
        assert all(checks["filters_matter"])
        assert checks["arithmetic_matters"] is (
            True if family != "count" else None
        )
        assert checks["grouping_matters"] is (
            True if family == "grouped_arithmetic" else None
        )
        assert task.score_answer(entry.answer, entry) == 1


def test_table_qa_difficulty_increases_semantic_complexity():
    easy = TableQAConfig()
    hard = TableQAConfig()
    easy.set_level(0)
    hard.set_level(4)

    assert hard.complexity > easy.complexity
    assert hard.num_tables == easy.num_tables == 1
