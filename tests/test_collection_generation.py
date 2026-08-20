import json

from easydict import EasyDict as edict

from reasoning_core import get_task, list_tasks, match_task_name, score_answer
from reasoning_core.generation_worker import run_task, serialize_example


def test_collection_adapters_are_explicit_and_do_not_change_default_tasks():
    for name in ("procedural_warmup", "reasoning_gym", "synlogic"):
        assert match_task_name(name) == name
        assert name not in list_tasks()


def test_collection_row_uses_individual_task_and_collection_metadata():
    example = get_task("procedural_warmup").generate_example(max_tokens=0, task="reverse")
    row = serialize_example(example)
    metadata = json.loads(row["metadata"])

    assert row["task"] == "reverse"
    assert metadata["source_collection"] == "procedural_warmup"
    assert metadata["source_task"] == row["task"]
    assert metadata["_task"] == "procedural_warmup"
    assert score_answer(example.answer, example) == 1


def test_procedural_warmup_reports_uncapped_effective_level():
    example = get_task("procedural_warmup").generate_example(level=3, max_tokens=0, task="reverse")

    assert example.metadata._level == 3
    assert example.metadata.effective_level == 3
    assert example.metadata.max_level is None


def test_generation_worker_writes_individual_collection_task(tmp_path):
    success, message = run_task("procedural_warmup", 0, 0, tmp_path, 1, 0)
    row = json.loads((tmp_path / "procedural_warmup-0.jsonl").read_text().splitlines()[0])

    assert (success, message) == (True, "OK")
    assert row["task"] == json.loads(row["metadata"])["source_task"]


def test_generation_worker_does_not_publish_partial_file(tmp_path, monkeypatch):
    class BadTask:
        timeout = None

        def generate_balanced_batch(self, **kwargs):
            return [edict(to_dict=lambda: {'metadata': {'bad': object()}})]

    monkeypatch.setattr('reasoning_core.generation_worker.get_task', lambda _: BadTask())
    success, message = run_task("bad", 0, 0, tmp_path, 1, 0)

    assert not success and message.startswith("ERR: TypeError:")
    assert not list(tmp_path.iterdir())


def test_native_row_keeps_its_task():
    example = get_task("arithmetics").generate_example(max_tokens=0)
    row = serialize_example(example)

    assert row["task"] == "arithmetics"
