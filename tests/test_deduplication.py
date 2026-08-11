import json
import random
import struct

import pandas as pd

from reasoning_core.tasks.binding import _mgu_rename, _mgu_split_key
from reasoning_core.tasks.formal_analogies import _canonical_case_structure
from reasoning_core.tasks.table_qa import canonical_table_pair
from reasoning_core.template import Config, Entry, Task, render_payload
from scripts.rc_preprocess_upload import deduplication_hash


class PayloadDedupTask(Task):
    def __init__(self, payload, answer="ok"):
        super().__init__(Config())
        self.payload = payload
        self.answer = answer

    def generate_entry(self):
        return Entry({"payload": self.payload}, self.answer)

    def render_prompt(self, metadata):
        return f"Static instructions.\n\n{render_payload(metadata.payload)}"


class SemanticDedupTask(PayloadDedupTask):
    def deduplication_key(self, problem):
        return ("same", True)


def test_default_key_is_stored_and_normalizes_only_shallow_payload_order():
    a = PayloadDedupTask({"left": "A", "right": "B"}).generate_example(max_tokens=0)
    b = PayloadDedupTask({"right": "B", "left": "A"}).generate_example(max_tokens=0)

    assert a.prompt != b.prompt
    assert a.deduplication_key == b.deduplication_key
    assert len(a.metadata["_deduplication_key"]) == 32


def test_default_key_includes_answer_and_override_is_hashed():
    a = PayloadDedupTask({"x": "1"}, answer="yes").generate_example(max_tokens=0)
    b = PayloadDedupTask({"x": "1"}, answer="no").generate_example(max_tokens=0)
    c = SemanticDedupTask({"x": "1"}, answer="yes").generate_example(max_tokens=0)

    assert a.deduplication_key != b.deduplication_key
    assert c.deduplication_key == ("same", True)
    assert len(c.metadata["_deduplication_key"]) == 32


def test_table_pair_ignores_sides_rows_columns_and_surface_scalars():
    a = pd.DataFrame({"n": [1000.0, None], "d": ["x", "y"]})
    b = pd.DataFrame({"d": ["y", "x"], "n": [None, 1000]})

    assert canonical_table_pair(a, b, "Yes") == canonical_table_pair(b, a, "Yes")


def test_analogical_structure_ignores_names_order_and_declared_direction_flips():
    first = _canonical_case_structure(
        [("r", "a", "b"), ("s", "b", "c")], ("r", "c", "a")
    )
    renamed_reordered_reversed = _canonical_case_structure(
        [("v", "y", "z"), ("u", "y", "x")], ("u", "x", "z")
    )

    assert first == renamed_reordered_reversed


def test_unification_key_preserves_sharing_but_ignores_declared_surface_changes():
    equations = [
        (("f", "x0"), ("f", "a")),
        (("h", "x0", "x1"), ("h", "a", "b")),
        ("x1", "b"),
    ]
    candidate = (("h", "x0", "x1"), ("h", "a", "b"))
    renamed_equations, renamed_candidate = _mgu_rename(equations, candidate, random)
    renamed_equations = [equation[::-1] for equation in reversed(renamed_equations)]

    assert _mgu_split_key(equations, candidate, "yes") == _mgu_split_key(
        renamed_equations, renamed_candidate[::-1], "yes"
    )
    assert _mgu_split_key([], (("h", "x0", "x0"), "a"), "no") != _mgu_split_key(
        [], (("h", "x0", "x1"), "a"), "no"
    )


def test_upload_prefers_metadata_key_and_fallback_keeps_distinct_answers():
    key = "0123456789abcdef0123456789abcdef"
    row = {"prompt": "p", "answer": "a", "metadata": json.dumps({"_deduplication_key": key})}

    assert deduplication_hash(row) == struct.unpack("<QQ", bytes.fromhex(key))
    assert deduplication_hash({"prompt": "p", "answer": "a", "metadata": "{}"}) != deduplication_hash(
        {"prompt": "p", "answer": "b", "metadata": "{}"}
    )
