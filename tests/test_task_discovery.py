import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import reasoning_core
import pytest
from reasoning_core import _discover_tasks
from reasoning_core import registry


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

    monkeypatch.setattr(
        registry,
        "_task_maps",
        lambda: ({"core": ("core", "Core"), "mutated_example": ("mutated.x", "Mutated")}, {}),
    )
    assert reasoning_core.list_tasks() == ["core"]
    assert reasoning_core.list_tasks(include_mutated=True) == ["core", "mutated_example"]


@pytest.mark.parametrize("statement", ["import reasoning_core", "import reasoning_core.template"])
def test_import_does_not_scan_task_tree(statement):
    code = (
        "from pathlib import Path\n"
        "def fail(*args, **kwargs): raise RuntimeError('task scan during import')\n"
        "Path.rglob = fail\n"
        f"{statement}\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=TASKS_ROOT.parents[1])


def test_list_tasks_discovers_once_per_process(monkeypatch):
    calls = 0
    original = registry._discover_tasks

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(registry, "_discover_tasks", counted)
    registry._task_maps.cache_clear()
    try:
        reasoning_core.list_tasks()
        reasoning_core.list_tasks()
        assert calls == 1
    finally:
        registry._task_maps.cache_clear()


def test_get_task_imports_only_its_module():
    code = (
        "import sys, reasoning_core\n"
        "reasoning_core.get_task('arithmetics')\n"
        "loaded = {name for name in sys.modules if name.startswith('reasoning_core.tasks.')}\n"
        "assert loaded == {'reasoning_core.tasks.arithmetics'}, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=TASKS_ROOT.parents[1])


def test_incremental_discovery_cache_reparses_only_changed_files(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    root.mkdir()
    first = root / "first.py"
    second = root / "second.py"
    first.write_text("from x import Task\nclass First(Task): pass\n")
    second.write_text("from x import Task\nclass Second(Task): pass\n")
    cache = tmp_path / "registry.json"
    parsed = []
    original = registry._parse_task_file

    def counted(path, relative):
        parsed.append(relative.as_posix())
        return original(path, relative)

    monkeypatch.setattr(registry, "_parse_task_file", counted)
    registry._discover_tasks(root, cache_path=cache)
    assert parsed == ["first.py", "second.py"]

    parsed.clear()
    registry._discover_tasks(root, cache_path=cache)
    assert parsed == []

    stat = first.stat()
    first.write_text("from x import Task\nclass Other(Task): pass\n")
    os.utime(first, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    parsed.clear()
    tasks, _ = registry._discover_tasks(root, cache_path=cache)
    assert parsed == ["first.py"]
    assert set(tasks) == {"other", "second"}

    second.unlink()
    registry._discover_tasks(root, cache_path=cache)
    assert set(json.loads(cache.read_text())["files"]) == {"first.py"}


def test_discovery_supports_multiple_tasks_per_file(tmp_path):
    (tmp_path / "pair.py").write_text(
        "from x import Task\n"
        "class First(Task): pass\n"
        "class Second(Task):\n    task_name = 'renamed'\n"
    )

    tasks, _ = registry._discover_tasks(tmp_path)

    assert tasks == {"first": ("pair", "First"), "renamed": ("pair", "Second")}


def test_discovery_rejects_duplicate_task_names(tmp_path):
    (tmp_path / "one.py").write_text("from x import Task\nclass Same(Task): pass\n")
    (tmp_path / "two.py").write_text("from x import Task\nclass Same(Task): pass\n")

    with pytest.raises(RuntimeError, match="Duplicate task 'same'.*one.py:Same.*two.py:Same"):
        registry._discover_tasks(tmp_path)


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
