"""The set of registered tasks is a deliberate choice, not whatever is on disk.

`_discover_tasks` rglobs the tasks tree, so an untracked scratch directory joins
DATASETS -- and therefore any fresh pool build -- in silence. That happened once
(18 probe tasks). When this test fails, either the drift is unintended or the
manifest needs updating in the same commit as the task.
"""
import pathlib

import reasoning_core

MANIFEST = pathlib.Path(__file__).parent / "task_manifest.txt"


def test_registered_tasks_match_manifest():
    expected = set(MANIFEST.read_text().split())
    actual = set(reasoning_core.DATASETS)
    assert actual == expected, (
        f"unexpected tasks: {sorted(actual - expected)}; "
        f"missing tasks: {sorted(expected - actual)}")
