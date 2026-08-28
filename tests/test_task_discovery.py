import reasoning_core
from reasoning_core import _discover_tasks


def test_task_discovery_recurses_into_generated_modules(tmp_path):
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "example.py").write_text(
        "from reasoning_core.template import Task\n"
        "class NestedExample(Task):\n"
        "    task_name = 'generated_example'\n"
    )
    deprecated = tmp_path / "deprecated"
    deprecated.mkdir()
    (deprecated / "old.py").write_text(
        "from reasoning_core.template import DevTask\n"
        "class OldExample(DevTask):\n"
        "    pass\n"
    )

    tasks, dev_tasks = _discover_tasks(tmp_path)

    assert tasks == {"generated_example": ("generated.example", "NestedExample")}
    assert dev_tasks == {}


def test_mutated_tasks_are_discovered_but_hidden_by_default(tmp_path, monkeypatch):
    mutated = tmp_path / "mutated"
    mutated.mkdir()
    (mutated / "example.py").write_text(
        "from reasoning_core.template import Task\n"
        "class MutatedExample(Task):\n"
        "    task_name = 'mutated_example'\n"
    )

    tasks, _ = _discover_tasks(tmp_path)
    assert tasks == {"mutated_example": ("mutated.example", "MutatedExample")}

    monkeypatch.setattr(reasoning_core, "DATASETS", {"core": object(), "mutated_example": object()})
    monkeypatch.setattr(reasoning_core, "_mutated_task_names", {"mutated_example"})
    assert reasoning_core.list_tasks() == ["core"]
    assert reasoning_core.list_tasks(include_mutated=True) == ["core", "mutated_example"]
