import pandas as pd
import numpy as np
import pytest

from reasoning_core.tasks import table_qa


def test_partial_pearson_removes_shared_linear_signal():
    z = np.arange(12, dtype=float)
    residual = np.array([1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6], dtype=float)
    target = 10 * z + residual
    confounded = 3 * z
    associated = residual

    assert abs(table_qa.partial_pearson(target, associated, z[:, None])) > 0.99
    assert abs(table_qa.partial_pearson(target, confounded, z[:, None])) < 0.01


def test_renderer_distribution_is_compact_biased_but_varied():
    weights = table_qa.TABLE_RENDERER_WEIGHTS
    compact = {"to_csv", "to_tsv", "to_pipe", "to_string"}

    assert set(weights) == set(table_qa.get_renderers(pd.DataFrame()))
    assert sum(weights[name] for name in compact) == 55
    assert sum(weights[name] for name in compact) > sum(weights[name] for name in weights if name not in compact)
    assert all(weight > 0 for weight in weights.values())


def test_statistics_difficulty_is_a_bounded_continuous_knob():
    config = table_qa.TableStatisticsConfig()
    config.set_level(0)
    easy = (config.num_rows, config.num_numeric, config.num_categories, config.margin)
    config.set_level(4)
    hard = (config.num_rows, config.num_numeric, config.num_categories, config.margin)

    assert easy == (12, 4, 3, 0.45)
    assert hard[:3] == (20, 7, 5)
    assert 0.08 <= hard[3] < easy[3]


def test_sample_distinct_renderers_uses_weights_without_replacement(monkeypatch):
    calls = []

    def choose(population, weights, k):
        calls.append((list(population), list(weights), k))
        return [population[0]]

    monkeypatch.setattr(table_qa.random, "choices", choose)
    selected = table_qa.sample_distinct_renderers(["to_csv", "to_grid", "to_json"])

    assert selected == ["to_csv", "to_grid"]
    assert calls == [
        (["to_csv", "to_grid", "to_json"], [18, 2, 3], 1),
        (["to_grid", "to_json"], [2, 3], 1),
    ]


def test_standardized_mean_difference_is_symmetric_and_scaled():
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([2.0, 3.0, 4.0])

    assert table_qa.standardized_mean_difference(a, b) == pytest.approx(2.0)
    assert table_qa.standardized_mean_difference(a, b) == table_qa.standardized_mean_difference(b, a)


def test_signed_and_absolute_statistical_wording_is_precise():
    _, row = table_qa.gen_row_pearson(table_qa.TableStatisticsConfig())
    _, shift = table_qa.gen_distribution_shift(table_qa.TableStatisticsConfig())

    assert "highest Pearson correlation" in row["find"]
    assert "largest absolute standardized mean difference between G0 and G1" in shift["find"]
    assert shift["metric"] == "absolute standardized mean difference"


@pytest.mark.parametrize(
    ("generator", "score"),
    [
        (
            table_qa.gen_pearson_change,
            lambda df, c: abs(
                table_qa.partial_pearson(df["x0"], df[c], df[["x1"]])
                - table_qa.pearson(df["x0"], df[c])
            ),
        ),
        (
            table_qa.gen_group_robust_pearson,
            lambda df, c: min(
                table_qa.abs_pearson(part["x0"], part[c]) for _, part in df.groupby("group")
            ),
        ),
        (
            table_qa.gen_group_heterogeneity,
            lambda df, c: np.ptp([
                table_qa.pearson(part["x0"], part[c]) for _, part in df.groupby("group")
            ]),
        ),
        (
            table_qa.gen_distribution_shift,
            lambda df, c: table_qa.standardized_mean_difference(
                df.loc[df["group"] == "G0", c], df.loc[df["group"] == "G1", c]
            ),
        ),
    ],
)
def test_new_statistics_generators_select_the_measured_winner(generator, score):
    df, spec = generator(table_qa.TableStatisticsConfig())
    excluded = {"group", "x0", "x1"} if spec["family"] == "pearson_change" else {"group", "x0"}
    candidates = [c for c in df.columns if c not in excluded]
    if spec["family"] == "distribution_shift":
        candidates = [c for c in df.columns if c != "group"]

    assert spec["answer"] == max(candidates, key=lambda c: score(df, c))
    assert spec["margin"] >= table_qa.TableStatisticsConfig().margin


def test_statistics_answers_use_compact_identifiers():
    entry = table_qa.TableStatistics().generate_entry()

    assert entry.answer in table_qa.STAT_IDENTIFIERS
    assert len(entry.answer) == 1


def test_compact_identifiers_update_grouped_query_and_answer(monkeypatch):
    monkeypatch.setattr(table_qa.random, "sample", lambda population, count: list(population)[:count])
    df = pd.DataFrame({"group": ["G0"], "x0": [1], "x1": [2], "x2": [3]})
    renamed, spec = table_qa.compact_statistics_identifiers(df, {
        "family": "group_robust_pearson",
        "find": "strongest association with x0",
        "answer": "x2",
    })

    assert list(renamed.columns) == ["group", "A", "B", "C"]
    assert spec["find"] == "strongest association with A"
    assert spec["answer"] == "C"


@pytest.mark.parametrize(
    ("family", "df", "find", "answer", "expected_find", "expected_answer"),
    [
        (
            "column_pearson",
            pd.DataFrame({"x0": [1], "x1": [2], "x2": [3]}),
            "column name most associated with column x0",
            "x1",
            "column name most associated with column x1",
            "x2",
        ),
        (
            "row_pearson",
            pd.DataFrame({"row_id": ["R0", "R1", "R2"], "x0": [1, 2, 3]}),
            "row_id with highest Pearson correlation to row R0",
            "R1",
            "row_id with highest Pearson correlation to row R1",
            "R2",
        ),
        (
            "label_eta2",
            pd.DataFrame({"label": ["a"], "x0": [1], "x1": [2]}),
            "numeric column name most associated with column label",
            "x0",
            "numeric column name most associated with column label",
            "x1",
        ),
        (
            "categorical_nmi",
            pd.DataFrame({"label": ["a"], "c0": ["a"], "c1": ["b"]}),
            "categorical column name most associated with column label",
            "c0",
            "categorical column name most associated with column label",
            "c1",
        ),
    ],
)
def test_statistics_identifiers_are_permuted_with_query_and_answer(
    monkeypatch, family, df, find, answer, expected_find, expected_answer
):
    monkeypatch.setattr(
        table_qa.random,
        "sample",
        lambda population, count: list(population[1:]) + list(population[:1]),
    )
    renamed, spec = table_qa.permute_statistics_identifiers(
        df,
        {"family": family, "find": find, "answer": answer},
    )

    assert spec["find"] == expected_find
    assert spec["answer"] == expected_answer
    assert expected_answer in set(renamed.columns) | set(renamed.get("row_id", []))
