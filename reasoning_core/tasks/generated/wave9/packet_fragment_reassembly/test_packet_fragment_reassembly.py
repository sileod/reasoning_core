import random

import pytest

from reasoning_core.tasks.generated.wave9.packet_fragment_reassembly import packet_fragment_reassembly as mod


@pytest.fixture(autouse=True)
def _seed():
    random.seed(1234)
    yield


def test_generate_render_score_roundtrip():
    task = mod.PacketFragmentReassembly()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(20):
            e = task.generate_entry()
            prompt = task.render_prompt(e.metadata)
            assert mod._parse_answer(e.answer) is not None
            assert task.score_answer(e.answer, e) == 1.0
            assert prompt
            assert len(prompt) > 0


def test_score_rejects_junk():
    task = mod.PacketFragmentReassembly()
    e = task.generate_entry()
    assert task.score_answer("", e) == 0.0
    assert task.score_answer("not an answer", e) == 0.0
    assert task.score_answer(None, e) == 0.0


def test_answer_domain():
    for level in range(7):
        task = mod.PacketFragmentReassembly()
        task.config.set_level(level)
        for _ in range(30):
            e = task.generate_entry()
            if e.answer.startswith("UNRECOVERABLE:"):
                body = e.answer[len("UNRECOVERABLE:"):]
                assert body != ""
            else:
                assert mod._parse_answer(e.answer)[0] == "payload"


def test_verify_helper():
    task = mod.PacketFragmentReassembly()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(30):
            e = task.generate_entry()
            frags = [mod._Frag(f["index"], f["offset"], f["data"], f["missing"]) for f in e.metadata["fragments"]]
            placed = [(f.offset, f.offset + len(f.data)) for f in frags if not f.missing]
            L = e.metadata["length"]
            reass = mod._can_fully_reassemble(placed, L)
            assert reass == e.metadata["reassemblable"]
            assert mod._verify(e.metadata["payload"], placed, L, reass, e.answer)


def test_both_answer_kinds_present():
    kinds = set()
    task = mod.PacketFragmentReassembly()
    for level in range(7):
        task.config.set_level(level)
        for _ in range(50):
            e = task.generate_entry()
            kinds.add(mod._parse_answer(e.answer)[0])
    assert "payload" in kinds
    assert "error" in kinds


def test_metadata_json_serializable():
    import json
    task = mod.PacketFragmentReassembly()
    e = task.generate_entry()
    json.dumps(dict(e.metadata))
