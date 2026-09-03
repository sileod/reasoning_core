"""The parts of difficulty_tune that decide things, none of which need the API.

The model's proposal is the easy half to get right. What keeps the tool honest is what it
refuses: an incomplete curve, a patch that restates the defaults, and a patch that changes
nothing. Each test here is a mistake that was actually made and shipped a bad diff before
the check existed.
"""
import pytest

from reasoning_core.reports.difficulty_tune import (
    compile_method, config_source, curve, diagnose, splice, tune, verify)

MODEL = "m"


def cell(rate, n=8, format_ok=1.0, status="ok"):
    return {"status": status, "solve_rate": rate, "n": n, "format_ok": format_ok}


def test_diagnose_names_each_way_a_ladder_is_wrong():
    assert diagnose({0: 0.9, 3: 0.5, 6: 0.1}) is None
    assert diagnose({0: 0.95, 6: 0.88})[0] == "too-easy"
    assert diagnose({0: 0.10, 6: 0.00})[0] == "too-hard"
    assert diagnose({0: 0.10, 6: 0.70})[0] == "inverted"
    assert diagnose({0: 0.60, 6: 0.50})[0] == "flat"


def test_curve_reports_holes_rather_than_diagnosing_a_truncated_ladder():
    """graph_pathfinding measured 53/53/25/0 -- a fine ladder. The two hard cells fell below the
    sample floor, and the two that survived were equal, so it was diagnosed "flat" and bought a
    patch it did not need. Format failures and short batches cluster at the HARD end, so dropping
    cells one at a time truncates the curve from the top."""
    cache = {f"t|0|{MODEL}": cell(0.53), f"t|2|{MODEL}": cell(0.53),
             f"t|4|{MODEL}": cell(0.25, n=4), f"t|6|{MODEL}": cell(0.0, n=1)}
    points, holes = curve(cache, MODEL)["t"]
    assert points == {0: 0.53, 2: 0.53}
    assert set(holes) == {4, 6}                      # and the caller must skip, not diagnose
    assert diagnose(points)[0] == "flat"             # which is exactly the wrong answer


def test_curve_separates_models_and_drops_untrustworthy_cells():
    cache = {f"t|0|{MODEL}": cell(0.9), f"t|2|{MODEL}": cell(0.5, format_ok=0.3),
             f"t|4|{MODEL}": cell(None, status="format-fail"), "t|6|other": cell(0.1)}
    points, holes = curve(cache, MODEL)["t"]
    assert points == {0: 0.9}
    assert holes == {2: "format 30%", 4: "format-fail"}


def test_verify_rejects_a_method_that_restates_the_declared_defaults():
    """`self.max_count = 3 + level` scales correctly and silently ignores the field it copies."""
    cfg, _, _ = config_source("count_elements")
    method = compile_method(cfg, "def apply_difficulty(self, level):\n"
                                 "    self.max_count = 3 + level\n"
                                 "    self.list_size = 10 + 2 * level\n")
    ok, note = verify("count_elements", cfg, method, [0, 3])
    assert not ok and "hardcoded" in note


def test_verify_rejects_a_rewrite_that_changes_nothing():
    """`x += level` restated as `x = self.x + level`, with a rationale claiming it fixed a bug."""
    cfg, _, source = config_source("grid_navigation")
    method = compile_method(cfg, "def apply_difficulty(self, level):\n"
                                 "    self.n_objects = self.n_objects + level\n"
                                 "    self.grid = self.grid + level\n"
                                 "    self.n_steps = self.n_steps + level\n"
                                 "    self.n_rel = self.n_rel + level\n")
    ok, note = verify("grid_navigation", cfg, method, [0, 3])
    assert not ok and "no behavioural change" in note


def test_verify_rejects_a_method_that_ignores_level():
    cfg, _, _ = config_source("count_elements")
    method = compile_method(cfg, "def apply_difficulty(self, level):\n    pass\n")
    ok, note = verify("count_elements", cfg, method, [0, 3])
    assert not ok and "ignores `level`" in note


def test_too_hard_is_diagnosed_but_never_patched():
    """Level 0 IS the declared defaults, and apply_difficulty runs after they are restored, so it
    cannot lower them. Asking anyway bought a no-op with a confabulated explanation."""
    def explode(*a, **k):                      # no client call may happen on this path
        raise AssertionError("the model must not be asked to fix the base defaults")
    row = tune("grid_navigation", {0: 0.0, 6: 0.0}, explode, [0, 6])
    assert row["verdict"] == "too-hard" and "base field defaults" in row["error"]


def test_splicing_the_current_method_back_is_byte_identical():
    """The patch path must be a no-op when nothing changed, or --apply cannot be trusted."""
    _, path, source = config_source("count_elements")
    patched, text = splice(path, source, source)
    assert patched == path.read_text() and text == ""


def test_splice_refuses_when_the_method_is_not_uniquely_locatable():
    _, path, _ = config_source("count_elements")
    patched, why = splice(path, "def apply_difficulty(self, level):\n", "irrelevant")
    assert patched is None and "uniquely" in why
