from concurrent.futures import ThreadPoolExecutor

from reasoning_core.source_store import SourceStore, snapshot_parent


def test_same_source_has_same_id_and_round_trips_exactly(tmp_path):
    store = SourceStore(tmp_path / ".evolution" / "objects")
    source = "# coding: utf-8\nvalue = 'é'\n\n"

    first = store.put(source)
    second = store.put(source)

    assert first == second
    assert store.get(first) == source
    assert (store.objects_dir / first[:2] / f"{first}.py").is_file()


def test_parent_snapshot_survives_changes_and_rename(tmp_path):
    store = SourceStore(tmp_path / ".evolution" / "objects")
    parent = tmp_path / "parent.py"
    original = "class Parent:\n    pass\n"
    parent.write_text(original)

    parent_source, metadata = snapshot_parent(
        parent,
        idea="try a variant",
        hypothesis="H1",
        changes="change generation",
        generation={
            "provider_name": "openrouter",
            "model_name": "deepseek/example",
            "harness_name": "opencode",
            "harness_version": "1.18.20",
            "agent_name": "build",
            "settings": {"variant": "high"},
        },
        store=store,
    )
    parent.write_text("class Changed:\n    pass\n")
    parent.rename(tmp_path / "renamed.py")

    assert parent_source == original
    assert store.get(metadata["parent_source_id"]) == original
    assert metadata == {
        "parent_source_id": store.put(original),
        "idea": "try a variant",
        "hypothesis": "H1",
        "changes": "change generation",
        "generation": {
            "provider_name": "openrouter",
            "model_name": "deepseek/example",
            "harness_name": "opencode",
            "harness_version": "1.18.20",
            "agent_name": "build",
            "settings": {"variant": "high"},
        },
    }


def test_snapshot_requires_resolved_generation_identity(tmp_path):
    parent = tmp_path / "parent.py"
    parent.write_text("x = 1\n")

    try:
        snapshot_parent(
            parent,
            idea="variant",
            hypothesis="H1",
            changes="change x",
            generation={"model_name": "example"},
            store=SourceStore(tmp_path / "objects"),
        )
    except ValueError as error:
        assert "provider_name" in str(error)
        assert "harness_version" in str(error)
    else:
        raise AssertionError("incomplete generation metadata was accepted")


def test_concurrent_identical_put_is_atomic(tmp_path):
    store = SourceStore(tmp_path / ".evolution" / "objects")
    source = "x = 1\n" * 10_000

    with ThreadPoolExecutor(max_workers=8) as pool:
        source_ids = list(pool.map(store.put, [source] * 32))

    assert len(set(source_ids)) == 1
    assert store.get(source_ids[0]) == source
