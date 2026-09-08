from datasets import Dataset

# The Prime Intellect env moved out of the package to integrations/ and is deliberately NOT
# importable as reasoning_core.*: it ships its own pyproject and is not part of the distribution.
# Load it by path so this stays a test of the adapter rather than of the packaging layout.
import importlib.util as _ilu
import pathlib as _pathlib

_ENV = (_pathlib.Path(__file__).resolve().parents[1]
        / "integrations/primeintellect/reasoning_core_env/reasoning_core_env.py")
_spec = _ilu.spec_from_file_location("pi_reasoning_core_env", _ENV)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_filter_available_tasks = _mod._filter_available_tasks
_prepare_env_dataset = _mod._prepare_env_dataset
from reasoning_core.template import Config, DevTask, Problem


class MetadataProbeTask(DevTask):
    def generate(self):
        return Problem(metadata={}, answer="42")

    def prompt(self, metadata):
        return "What is the answer?"


MetadataProbeTask.__module__ = "reasoning_core.tests"


def test_generated_examples_include_agnostic_generator_metadata():
    example = MetadataProbeTask(Config()).generate_example(max_tokens=0)

    assert example.metadata._generator_name == "reasoning_core"
    assert example.metadata._generator_version
    assert "_generator_commit" in example.metadata
    assert example.metadata._task_version == "0"


def test_env_filter_ignores_unavailable_tasks():
    dataset = Dataset.from_list(
        [
            {"prompt": "keep", "answer": "1", "metadata": '{"_task": "available"}'},
            {"prompt": "drop", "answer": "2", "metadata": '{"_task": "deprecated"}'},
            {"prompt": "also keep", "answer": "3", "metadata": '{"_task": "available"}'},
        ]
    )

    filtered = _filter_available_tasks(dataset, available_tasks={"available"})

    assert len(filtered) == 2
    assert filtered["prompt"] == ["keep", "also keep"]


def test_env_dataset_drops_top_level_task_column():
    dataset = Dataset.from_list(
        [
            {
                "prompt": "keep",
                "answer": "1",
                "task": "available",
                "metadata": '{"_task": "available"}',
            },
        ]
    )

    prepared = _prepare_env_dataset(dataset, available_tasks={"available"})

    assert prepared.column_names == ["question", "answer", "info"]
    assert prepared[0]["question"] == "keep"
    assert prepared[0]["info"]["answer"] == "1"


if __name__ == "__main__":
    test_generated_examples_include_agnostic_generator_metadata()
    test_env_filter_ignores_unavailable_tasks()
    test_env_dataset_drops_top_level_task_column()
    print("generator metadata and env filter tests passed")
