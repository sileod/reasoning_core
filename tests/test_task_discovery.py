import ast
from pathlib import Path

import reasoning_core
from reasoning_core import _discover_tasks


TASKS_ROOT = Path(__file__).parents[1] / "reasoning_core" / "tasks"


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


def test_every_shipped_task_has_a_literal_coverage_summary():
    """The summary feeds the coverage table, so it is owed by whatever the table covers.

    Task-search output is discoverable before it is on the roster, and a landed wave is not
    yet in any table -- it earns the summary requirement when it earns its place, which is
    the same moment it joins `list_tasks()`.
    """
    tasks, dev_tasks = _discover_tasks(TASKS_ROOT)
    shipped = set(reasoning_core.list_tasks()) | set(dev_tasks)

    for name, (module_name, class_name) in {**tasks, **dev_tasks}.items():
        if name not in shipped:
            continue
        path = TASKS_ROOT.joinpath(*module_name.split(".")).with_suffix(".py")
        tree = ast.parse(path.read_text(), filename=str(path))
        node = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef) and node.name == class_name)
        assignment = next(
            (item for item in node.body
             if isinstance(item, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "summary"
                     for target in item.targets)),
            None,
        )
        assert assignment is not None, f"{module_name}.{class_name} needs summary"
        summary = ast.literal_eval(assignment.value)
        assert isinstance(summary, str) and summary.strip()
        assert summary == summary.strip() and "\n" not in summary and "\r" not in summary
