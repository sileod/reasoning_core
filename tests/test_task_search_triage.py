"""Triage picks one draft per idea, and knows unreviewed from reviewed-and-clean."""
import json

import pytest

from reasoning_core.task_search.triage import (
    CACHE_NAME,
    pick,
    proposal_of,
    successes,
    summarize,
)


def _trial(tmp_path, run, trial_id, status="success", verdict=None, name="a_task_v1",
           used=10, exhausted=False):
    directory = tmp_path / run / trial_id
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps({
        "trial_id": trial_id,
        "status": status,
        "sample_sanity": {"verdict": verdict, "why": "-" if verdict else "unreachable"},
        "steps": {"used": used, "max": 40, "exhausted": exhausted},
        "changed_paths": [f"reasoning_core/tasks/generated/w/{name}/{name}.py"],
    }))
    return directory


def test_a_retry_corpse_is_not_counted_as_a_second_success(tmp_path):
    """A provider 429 leaves `P003v2.attempt1-provider_429` beside the real trial."""
    _trial(tmp_path, "stamp", "P003v2")
    _trial(tmp_path, "stamp", "P003v2.attempt1-provider_429")

    found = successes(tmp_path)

    assert [directory.name for directory, _ in found] == ["P003v2"]


def test_only_successes_are_triaged(tmp_path):
    _trial(tmp_path, "stamp", "P001v1")
    _trial(tmp_path, "stamp", "P002v1", status="validation_failed")

    assert [trial["trial_id"] for _, trial in successes(tmp_path)] == ["P001v1"]


@pytest.mark.parametrize("verdicts, expected", [
    # A read-and-cleared draft beats one nobody could read, whichever came first.
    ((None, "VALID"), "P001v2"),
    (("VALID", None), "P001v1"),
    # Between two the reviewer cleared, the one that did not run out of budget wins.
    (("VALID", "VALID"), "P001v2"),
])
def test_the_better_reviewed_draft_of_an_idea_is_the_one_picked(verdicts, expected):
    drafts = [
        {"trial": "P001v1", "verdict": verdicts[0], "exhausted": True},
        {"trial": "P001v2", "verdict": verdicts[1], "exhausted": False},
    ]

    assert pick(drafts)[0]["trial"] == expected


def test_an_unreviewed_draft_is_reported_apart_from_a_cleared_one():
    """The gate fails open, so `semantics ok` alone cannot say a task was read.

    Counting an outage as a pass is right for the gate and wrong here: promotion is
    exactly where "nobody checked" must not be spelled the same as "checked and clean".
    """
    groups = {
        "P001": [{"trial": "P001v1", "verdict": "VALID", "exhausted": False}],
        "P002": [{"trial": "P002v1", "verdict": None, "exhausted": False}],
        "P003": [{"trial": "P003v1", "verdict": "INVALID", "exhausted": False}],
    }

    assert summarize(groups) == {"take": 1, "unreviewed": 1, "drop": 1, "gameable": 0}


def test_a_gameable_pick_is_withheld_however_cleanly_the_reviewer_read_it():
    """The in-trial audit runs at n=30, noisy right at the 0.40 ceiling.

    wave0's n02 cleared it at n=30 and lost at n=40; the wider n is the one that
    decides, and a semantic VALID says nothing about whether the answer can be guessed.
    """
    groups = {"P001": [{"trial": "P001v1", "verdict": "VALID", "exhausted": False,
                        "audit": {"ok": False, "why": "constant guess +0.45"}}]}

    assert summarize(groups)["gameable"] == 1
    assert summarize(groups)["take"] == 0


def test_a_cached_review_is_preferred_to_the_null_the_run_recorded(tmp_path):
    directory = _trial(tmp_path, "stamp", "P001v1", verdict=None)
    (directory / CACHE_NAME).write_text(json.dumps({"verdict": "INVALID", "why": "x"}))

    from reasoning_core.task_search.triage import _recorded_verdict

    trial = json.loads((directory / "run.json").read_text())
    verdict, source = _recorded_verdict(directory, trial)

    assert (verdict["verdict"], source) == ("INVALID", "triage")


def test_a_trial_id_rerun_in_a_later_run_is_the_one_taken():
    """Two runs of the same wave each produce a P008v1, and they are not the same task."""
    drafts = [
        {"trial": "P008v1", "run": "20260901T191036Z", "verdict": "VALID", "exhausted": False},
        {"trial": "P008v1", "run": "20260901T193327Z", "verdict": "VALID", "exhausted": False},
    ]

    assert pick(drafts)[0]["run"] == "20260901T193327Z"


def test_drafts_of_one_idea_group_together():
    assert proposal_of("P001v1") == proposal_of("P001v2") == "P001"
