from collections import Counter
from datetime import date

import numpy as np
import pandas as pd

import reasoning_core.tasks.table_qa as table_qa
from reasoning_core.tasks.table_qa import (
    Predicate, QueryPlan, TableEquivalence, TableQA, TableQAConfig,
    canonical_scalar, canonical_table, corrupt_table, equivalence_display,
    legal_extensions, render_nulls, render_query, sample_query_plan,
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


def test_query_plan_renders_topk_and_grouped_arithmetic():
    topk = QueryPlan(projection=["row_id"], order_by=["qty"], limit=3)
    grouped = QueryPlan(
        predicates=[Predicate("status", "paid")], projection=["category"],
        expression='"qty" * "unit_price"', aggregate="sum",
        group_by=["category"], order_by=["aggregate"], limit=1,
    )

    assert render_query(topk) == (
        'SELECT "row_id" FROM dataframe WHERE TRUE ORDER BY "qty" DESC LIMIT 3'
    )
    assert "GROUP BY" in render_query(grouped)
    assert "ORDER BY SUM" in render_query(grouped)
    assert "LIMIT 1" in render_query(grouped)


def test_table_qa_conditions_every_sampled_operator():
    task = TableQA(TableQAConfig(num_rows=12, column_slack=1.5, complexity=4))

    for _ in range(20):
        entry = task.generate_entry()
        spec = entry.metadata["query_spec"]
        assert spec["query_conditioned"]
        assert all(spec["feature_checks"].values())
        assert len(entry.answer.splitlines()) <= 6
        assert task.score_answer(entry.answer, entry) == 1


def test_query_plan_complexity_is_continuous_and_compositional():
    low = [sample_query_plan(TableQAConfig(complexity=0.5)) for _ in range(300)]
    high = [sample_query_plan(TableQAConfig(complexity=8.0)) for _ in range(300)]
    density = lambda plans: np.mean([
        len(p.predicates) + bool(p.expression) + bool(p.aggregate) + bool(p.group_by)
        + len(p.order_by) + bool(p.limit) + p.distinct for p in plans
    ])

    assert density(high) > density(low) + 3
    assert any(p.limit for p in high)
    assert any(p.group_by for p in high)
    assert "group" not in legal_extensions(QueryPlan(projection=["row_id"]))


def test_table_qa_difficulty_increases_semantic_complexity():
    easy = TableQAConfig()
    hard = TableQAConfig()
    easy.set_level(0)
    hard.set_level(4)

    assert hard.complexity > easy.complexity
    assert hard.num_rows > easy.num_rows
    assert hard.column_slack > easy.column_slack


def test_table_qa_task_version_is_five():
    entry = TableQA(TableQAConfig(complexity=0)).generate_example(max_tokens=0)
    assert entry.metadata._task_version == 5
