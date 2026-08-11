from scripts.benchmark_task_scaling import bounded_cell, parse_levels


def test_parse_levels_accepts_ranges_and_lists():
    assert parse_levels("0-2,4,6") == [0, 1, 2, 4, 6]


def test_benchmark_cell_reports_level_support():
    row = bounded_cell("arithmetics", level=6, samples=2, max_tokens=8192, timeout=30)

    assert row["status"] == "supported"
    assert row["successes"] == row["valid_scores"] == 2
    assert row["wall_seconds_mean"] >= 0
    assert row["wall_seconds_first"] >= 0
    assert row["wall_seconds_warm_mean"] >= 0
    assert row["prompt_tokens_mean"] > 0
    assert row["config"]["level"] == 6
