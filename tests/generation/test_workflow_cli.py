import json
import subprocess
import sys

import pytest

from reasoning_core import registry
from reasoning_core.__main__ import main


def test_catalog_search_uses_source_without_importing(tmp_path, monkeypatch, capsys):
    (tmp_path / "graph.py").write_text(
        "raise RuntimeError('must not import')\n"
        "class Graph(Task):\n    summary = 'Find a shortest path in a graph.'\n"
    )
    (tmp_path / "generated").mkdir()
    (tmp_path / "generated" / "extra.py").write_text("class Extra(Task):\n    pass\n")
    monkeypatch.setattr(registry, "_TASKS_PATH", tmp_path)
    monkeypatch.setattr(registry, "_task_maps", lambda: (
        {"graph": ("graph", "Graph"), "extra": ("generated.extra", "Extra")}, {}))
    main(["catalog", "PATH graph", "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert rows == [dict(name="graph", summary="Find a shortest path in a graph.",
                         status="active", origin="core", source="reasoning_core/tasks/graph.py", line=2)]
    assert len(registry.task_catalog()) == 1
    assert len(registry.task_catalog(include_generated=True)) == 2
    assert registry.task_catalog("nonexistent") == []


def test_sample_score_roundtrip_and_no_overwrite(tmp_path, capsys, monkeypatch):
    import xxhash
    original_hash = xxhash.xxh3_128_hexdigest

    def bytes_only(value):
        assert isinstance(value, bytes)  # xxhash 4 no longer implicitly encodes strings.
        return original_hash(value)

    monkeypatch.setattr(xxhash, "xxh3_128_hexdigest", bytes_only)
    path = tmp_path / "samples.jsonl"
    main(["sample", "arithmetics", "--count", "2", "--output", str(path)])
    capsys.readouterr()
    original = path.read_text()
    with pytest.raises(FileExistsError):
        main(["sample", "arithmetics", "--count", "1", "--output", str(path)])
    assert path.read_text() == original
    rows = [json.loads(line) for line in original.splitlines()]
    path.write_text("\n".join(json.dumps({**row, "prediction": row["answer"]}) for row in rows))
    main(["score", str(path)])
    assert json.loads(capsys.readouterr().out)["mean_score"] == 1


def test_score_rejects_missing_predictions(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{}\n')
    with pytest.raises(SystemExit) as error:
        main(["score", str(path)])
    assert error.value.code == 2


def test_collection_import_preserves_process_configuration():
    pytest.importorskip("pyarrow")
    code = '''
import os, sys, tempfile
import huggingface_hub, nfsdict, tqdm, pyarrow.parquet
before = dict(os.environ), tempfile.tempdir
sys.argv = ['host-application', '--unrelated-option']
import reasoning_core.generation.collect
assert (dict(os.environ), tempfile.tempdir) == before
'''
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
