from reasoning_core.tasks._synlogic import Synlogic, SynlogicConfig


def _task(level):
    task = object.__new__(Synlogic)
    task.config = SynlogicConfig(seed=0).set_level(level)
    return task


def test_native_synlogic_difficulty_tracks_level_and_saturates():
    assert _task(0)._generate_kwargs("sudoku")["difficulty"] == 1
    assert _task(2)._generate_kwargs("sudoku")["difficulty"] == 3
    assert _task(5)._generate_kwargs("sudoku")["difficulty"] == 4
    assert _task(5)._generate_kwargs("web_of_lies")["difficulty"] == 5
    assert _task(5).config.difficulty == 5


def test_synlogic_games_without_native_difficulty_keep_source_defaults():
    assert "difficulty" not in _task(3)._generate_kwargs("boolean_expressions")


def test_synlogic_reports_effective_and_max_level():
    example = Synlogic(SynlogicConfig(task="web_of_lies", language="mixed")).generate_example(
        level=5, max_tokens=0
    )

    assert example.metadata._level == 5
    assert example.metadata.effective_level == 4
    assert example.metadata.max_level == 4
