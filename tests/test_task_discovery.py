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
