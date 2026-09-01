import json
from pathlib import Path

import pytest
import yaml

from reasoning_core.task_search.wave_proposer import (
    CatalogEntry,
    ChatClient,
    _extract_json,
    build_catalog,
    catalog_record,
    check_proposal_file,
    closest_entries,
    proposal_problems,
    propose_wave,
    validate_proposal_wave,
    write_proposal_wave,
)


ROOT = Path(__file__).parents[1]


def proposal(name="fresh_operation"):
    return {
        "name": name,
        "summary": ("Propagate signed relation constraints across cycles and overlapping"
                    " paths, answering the queried pair's parity."),
    }


class FakeClient:
    def __init__(self, generated, reviews):
        self.generated = generated
        self.reviews = reviews
        self.calls = []

    def json(self, purpose, system, user, max_tokens=32768):
        self.calls.append({"purpose": purpose, "request_sha256": "a",
                           "response_sha256": "b", "response_id": purpose})
        return self.generated if purpose.startswith("propose") else self.reviews


def test_catalog_includes_gallery_plans_and_tasks():
    entries = build_catalog(ROOT)
    sources = catalog_record(entries)["sources"]

    assert sources["gallery"] >= 60
    assert sources["plan"] >= 100
    assert sources["task"] >= 60
    assert any(entry.name == "graph_pathfinding" for entry in entries)


def test_catalog_uses_task_coverage_summaries_for_novelty():
    entries = {entry.entry_id: entry for entry in build_catalog(ROOT)}

    csp = entries["task:constraint_satisfaction:ConstraintSatisfaction"]
    arithmetic = entries["task:arithmetics:Arithmetics"]
    assert csp.summary == (
        "Solve query-aware assignment, graph, scheduling, grid, set, and numeric CSPs.")
    assert arithmetic.summary == (
        "Compositional arithmetics with float/int/bool, varied operators, number theory.")


def test_closest_entries_uses_signature_not_only_name():
    catalog = (
        CatalogEntry("old:parity", "signed_graph_query",
                     "compose signed constraints and output queried parity", "plan"),
        CatalogEntry("old:sort", "sorting", "sort a list", "task"),
    )

    assert closest_entries(proposal(), catalog, limit=1)[0].entry_id == "old:parity"


def test_a_proposal_is_a_name_and_a_coverage_summary():
    assert proposal_problems(proposal()) == []

    assert proposal_problems({"name": "Bad Name", "summary": proposal()["summary"]}) == [
        "name must be canonical snake_case"]
    assert proposal_problems({"name": "ok_task", "summary": "sorts a list"}) == [
        "summary must be 40-240 characters"]
    assert proposal_problems({"name": "ok_task", "summary": "x " * 40}) == [
        "summary must be one trimmed line"]


def test_a_proposal_may_not_smuggle_back_the_old_boilerplate():
    """Difficulty ladders are a library-wide choice, not a per-proposal one."""
    noisy = proposal()
    noisy["data"] = {"difficulty": {"level_0": "one path"}}
    noisy["demonstration"] = {"prompt": "...", "answer": "..."}

    assert proposal_problems(noisy) == ["unexpected keys: data, demonstration"]


def test_proposer_rejects_exact_catalog_collision_then_accepts_critic_novelty():
    duplicate = proposal("graph_pathfinding")
    fresh = proposal("signed_constraint_parity")
    client = FakeClient(
        {"proposals": [duplicate, fresh]},
        {"reviews": [{
            "proposal_id": "C001", "verdict": "novel",
            "nearest_neighbors": [
                {"id": "plan:WAVE0:M1", "relationship": "adjacent",
                 "overlap": "both propagate constraints"},
                {"id": "gallery:belief_tracking", "relationship": "different",
                 "overlap": "both update latent relations"},
                {"id": "gallery:constraint_satisfaction", "relationship": "adjacent",
                 "overlap": "both enforce consistency"},
            ],
            "substantive_difference": "tracks parity over redundant signed paths",
            "scores": {"novelty": 5, "sft_value": 5, "feasibility": 4, "clarity": 4},
            "reason": "The cognitive operation is distinct.",
        }]},
    )

    wave = propose_wave(ROOT, name="unit-wave", count=1, rounds=1, client=client)

    assert wave["proposals"][0]["name"] == "signed_constraint_parity"
    assert wave["proposals"][0]["id"] == "P001"
    assert wave["proposals"][0]["novelty"]["verdict"] == "novel"
    assert len(wave["proposals"][0]["novelty"]["nearest_neighbors"]) == 3
    assert wave["rejected"][0]["verdict"] == "duplicate"
    assert validate_proposal_wave(wave) == []


def test_archive_is_durable_and_never_overwritten(tmp_path):
    item = proposal()
    item.update({"id": "P001", "novelty": {
        "verdict": "novel", "substantive_difference": "new operation",
        "scores": {"novelty": 5, "sft_value": 5, "feasibility": 4, "clarity": 4},
        "nearest_neighbors": [
            {"id": f"old:{i}", "relationship": "different", "overlap": "same family"}
            for i in range(3)
        ],
    }})
    wave = {"format_version": 1, "kind": "sft_task_proposals",
            "name": "x", "proposals": [item]}
    path = tmp_path / "archive" / "x.yaml"

    write_proposal_wave(path, wave)

    assert check_proposal_file(path) == []
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_proposal_wave(path, wave)


def test_json_extraction_accepts_fenced_reasoning_output():
    assert _extract_json("analysis first\n```json\n{\"proposals\": []}\n```\n") == {
        "proposals": []}


def test_incomplete_wave_is_returned_so_model_calls_are_not_lost():
    client = FakeClient({"proposals": [proposal("graph_pathfinding")]}, {"reviews": []})

    wave = propose_wave(ROOT, name="partial-wave", count=2, rounds=1, client=client)

    assert wave["objective"] == {
        "training_stage": "sft", "requested": 2, "accepted": 0, "complete": False}
    assert wave["rejected"][0]["verdict"] == "duplicate"


def test_neighbor_evidence_overrides_an_inconsistent_novel_verdict():
    client = FakeClient(
        {"proposals": [proposal("signed_constraint_parity")]},
        {"reviews": [{
            "proposal_id": "C001", "verdict": "novel",
            "nearest_neighbors": [
                {"id": "gallery:constraint_satisfaction", "relationship": "variant",
                 "overlap": "same constraint propagation operation"},
                {"id": "gallery:belief_tracking", "relationship": "adjacent",
                 "overlap": "both update latent relations"},
                {"id": "gallery:logic_derivation", "relationship": "different",
                 "overlap": "both compose multiple steps"},
            ],
            "substantive_difference": "only changes relation labels",
            "scores": {"novelty": 5, "sft_value": 5, "feasibility": 5, "clarity": 5},
            "reason": "The proposal is useful.",
        }]},
    )

    wave = propose_wave(ROOT, name="neighbor-gate", count=1, rounds=1, client=client)

    assert wave["proposals"] == []
    assert "contradicted" in wave["rejected"][0]["reason"]


def test_check_proposals_reports_a_missing_archive(tmp_path):
    assert check_proposal_file(tmp_path / "missing.yaml") == [
        f"proposal wave does not exist: {tmp_path / 'missing.yaml'}"]


def test_client_polls_a_pending_asynchronous_request(monkeypatch):
    class Response:
        def __init__(self, status, payload):
            self.status_code = status
            self._payload = payload
            self.content = json.dumps(payload).encode()

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

    posted = Response(202, {"requestId": "abc"})
    finished = Response(200, {"id": "response-1", "choices": [{"message": {
        "content": '{"proposals": []}'}}]})
    observed = {}
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: posted)
    monkeypatch.setattr("time.sleep", lambda _: None)

    def get(url, **kwargs):
        observed["url"] = url
        return finished

    monkeypatch.setattr("requests.get", get)
    client = ChatClient(api_key="secret", timeout=10)

    assert client.json("test", "system", "user") == {"proposals": []}
    assert observed["url"] == "https://integrate.api.nvidia.com/v1/status/abc"
    assert client.calls[0]["response_id"] == "response-1"
    assert "secret" not in json.dumps(client.calls)
