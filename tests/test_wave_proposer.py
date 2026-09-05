import json
import re
from pathlib import Path

import pytest
import yaml

from reasoning_core.task_search.wave_proposer import (
    CatalogEntry,
    CRITIC_MAX_BATCH,
    ChatClient,
    UpstreamError,
    _critic_votes,
    _extract_json,
    _proposal_entries,
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
VARIANT = re.compile(r"_v\d+$")


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
    assert sources["plan"] >= 90
    # Not a floor on proposals. The catalog keys by name and prefers the best account of
    # an idea, so a proposal that got built is counted as the task it became: this number
    # falls as the pipeline succeeds, and was 80 back when most of the catalog was still
    # unbuilt. What has to hold is that no proposed idea drops out of the catalog, which
    # is what the catalog is for -- a wave is remembered even if nobody implemented it.
    assert sources["proposal"] >= 20
    assert {entry.name for entry in _proposal_entries(ROOT)} <= {entry.name for entry in entries}
    assert any(entry.name == "graph_pathfinding" for entry in entries)


def test_the_catalog_names_each_idea_once():
    """The catalog is a prompt, so a repeat is paid for on every proposal call.

    Each shipped task was also a gallery line, each `external` proposal was also two
    wave8 plan trials, and the trials arrived under names -- `..._v1`, `..._v2` -- that
    no proposal would ever collide with. That was 499 entries for 278 ideas.
    """
    entries = build_catalog(ROOT)
    names = [entry.name for entry in entries]

    assert len(names) == len(set(names))
    assert not [name for name in names if VARIANT.search(name)]
    assert not any(entry.entry_id.startswith("plan:wave8:") for entry in entries), (
        "wave8 fanned the external eighty into drafts; the ideas are already catalogued")


def test_catalog_uses_task_coverage_summaries_for_novelty():
    # By name, not by entry id: a shipped task is also a gallery line, the catalog now
    # keeps one of the two, and the summary is the same either way -- which is the point.
    entries = {entry.name: entry for entry in build_catalog(ROOT)}

    csp = entries["constraint_satisfaction"]
    arithmetic = entries["arithmetics"]
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
        "summary must be 40-240 characters, got 12"]
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


def test_client_reads_a_streamed_reply(monkeypatch):
    """Streaming is the default because a silent request to NIM is killed by the gateway.

    kimi-k3 spends minutes reasoning before its first content token, and wave9 died on a
    504 at batch 12 and again at batch 6 while a 16-token request to the same model
    answered fine. The bytes are what hold the connection open, so the reader has to
    survive keep-alive blanks, reasoning-only deltas and the terminating sentinel.
    """
    chunks = [
        'data: {"id": "stream-1", "choices": [{"delta": {"reasoning_content": "think"}}]}',
        "",
        'data: {"id": "stream-1", "choices": [{"delta": {"content": "{\\"proposals\\":"}}]}',
        ": keep-alive",
        'data: {"id": "stream-1", "choices": [{"delta": {"content": " []}"}}]}',
        "data: [DONE]",
        'data: {"id": "stream-1", "choices": [{"delta": {"content": "ignored"}}]}',
    ]

    class Response:
        status_code = 200
        content = b"consuming this would eat the stream"

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=False):
            return iter(chunks)

    observed = {}

    def post(url, **kwargs):
        observed.update(stream=kwargs.get("stream"), body=kwargs.get("json"))
        return Response()

    monkeypatch.setattr("requests.post", post)
    client = ChatClient(api_key="secret", timeout=10)

    assert client.json("test", "system", "user") == {"proposals": []}
    # Asking for a stream and then not reading it as one is the failure that hangs.
    assert observed["stream"] is True and observed["body"]["stream"] is True
    assert client.calls[0]["response_id"] == "stream-1"
    assert "secret" not in json.dumps(client.calls)


def test_a_client_can_still_be_asked_for_a_whole_document(monkeypatch):
    """Endpoints that do not stream have to keep working, so the flag stays a choice."""
    class Response:
        status_code = 200
        content = json.dumps({"id": "doc-1", "choices": [
            {"message": {"content": '{"proposals": []}'}}]}).encode()

        def raise_for_status(self):
            return None

        def json(self):
            return json.loads(self.content)

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: Response())
    client = ChatClient(api_key="secret", timeout=10, stream=False)

    assert client.json("test", "system", "user") == {"proposals": []}
    assert client.calls[0]["response_id"] == "doc-1"


def test_the_critic_can_run_on_a_separate_client_from_the_proposer(tmp_path):
    """A round is two calls, and wave9 lost a whole wave to a 429 raised by the second.

    The proposer had already spent the shared quota by the time the critic asked for its
    turn, so the split is what keeps one call from starving the other. The archive has to
    say which model actually issued the novelty verdicts, or a wave's acceptances cannot
    be attributed later.
    """
    class Client:
        def __init__(self, model, reply):
            self.model, self.provider, self.endpoint = model, "fake", "http://fake"
            self.reasoning_effort, self.calls, self.purposes = None, [], []
            self._reply = reply

        def json(self, purpose, system, user):
            self.purposes.append(purpose)
            self.calls.append({"purpose": purpose})
            return self._reply

    proposer = Client("big", {"proposals": [
        {"name": "novel_thing", "summary": 'Given a labelled hypergraph and a rewrite budget, decide which contraction orders reach the target normal form and report the cheapest one.'}]})
    critic = Client("small", {"reviews": [{
        "proposal_id": "C001", "verdict": "novel",
        "nearest_neighbors": [
            {"id": "plan:WAVE0:M1", "relationship": "adjacent",
             "overlap": "both propagate constraints"},
            {"id": "gallery:belief_tracking", "relationship": "different",
             "overlap": "both update latent relations"},
            {"id": "gallery:constraint_satisfaction", "relationship": "adjacent",
             "overlap": "both enforce consistency"},
        ],
        "substantive_difference": "contracts hyperedges under a cost budget",
        "scores": {"novelty": 5, "sft_value": 5, "feasibility": 5, "clarity": 5},
        "reason": "distinct operation"}]})

    wave = propose_wave(ROOT, name="split", count=1, rounds=1,
                        client=proposer, critic_client=critic)

    assert [p["name"] for p in wave["proposals"]] == ["novel_thing"]
    assert all("propose-round" in purpose for purpose in proposer.purposes)
    assert all("critic-round" in purpose for purpose in critic.purposes)
    assert wave["review"]["model"] == "small"
    assert wave["review"]["shared_with_generator"] is False
    assert wave["generation"]["calls"] and wave["review"]["calls"]


def test_one_client_still_does_both_jobs_when_no_critic_is_given(tmp_path):
    """The split is opt-in at the library level, so existing callers keep working."""
    class Client:
        model, provider, endpoint = "solo", "fake", "http://fake"
        reasoning_effort = None

        def __init__(self):
            self.calls, self.purposes = [], []

        def json(self, purpose, system, user):
            self.purposes.append(purpose)
            self.calls.append({"purpose": purpose})
            if purpose.startswith("propose"):
                return {"proposals": [{"name": "solo_thing", "summary": 'Given a labelled hypergraph and a rewrite budget, decide which contraction orders reach the target normal form and report the cheapest one.'}]}
            return {"reviews": [{
                "proposal_id": "C001", "verdict": "novel",
                "substantive_difference": "it is different",
                "scores": {"novelty": 5, "sft_value": 5, "feasibility": 5, "clarity": 5},
                "nearest_neighbors": [], "reason": "distinct operation"}]}

    solo = Client()
    wave = propose_wave(ROOT, name="solo", count=1, rounds=1, client=solo)

    assert wave["review"]["shared_with_generator"] is True
    assert wave["review"]["calls"] == []
    assert len(solo.purposes) == 2


def _neighbors():
    return [
        {"id": "plan:WAVE0:M1", "relationship": "adjacent",
         "overlap": "both propagate constraints"},
        {"id": "gallery:belief_tracking", "relationship": "different",
         "overlap": "both update latent relations"},
        {"id": "gallery:constraint_satisfaction", "relationship": "adjacent",
         "overlap": "both enforce consistency"},
    ]


class VotingCritic:
    """A critic whose verdict per sample is scripted, one entry per call."""

    model, provider, endpoint = "small", "fake", "http://fake"
    reasoning_effort = None

    def __init__(self, verdicts):
        self.verdicts, self.calls, self.purposes = list(verdicts), [], []

    def json(self, purpose, system, user):
        self.purposes.append(purpose)
        self.calls.append({"purpose": purpose})
        verdict = self.verdicts.pop(0)
        if verdict is None:
            return {"reviews": []}
        return {"reviews": [{
            "proposal_id": "C001", "verdict": verdict,
            "nearest_neighbors": _neighbors(),
            "substantive_difference": "contracts hyperedges under a cost budget",
            "scores": {"novelty": 5 if verdict == "novel" else 2, "sft_value": 5,
                       "feasibility": 5, "clarity": 5},
            "reason": f"judged {verdict}"}]}


def _one_proposal_client(*names):
    class Client:
        model, provider, endpoint = "big", "fake", "http://fake"
        reasoning_effort = None

        def __init__(self):
            self.calls = []

        def json(self, purpose, system, user):
            self.calls.append({"purpose": purpose})
            return {"proposals": [proposal(name) for name in names]}

    return Client()


@pytest.mark.parametrize("verdicts, accepted, votes", [
    (["novel", "novel", "duplicate"], True, "2/3"),
    (["novel", "duplicate", "duplicate"], False, "1/3"),
])
def test_the_critic_is_polled_k_times_and_the_majority_decides(verdicts, accepted, votes):
    """One flash opinion is noisy, and its two errors do not cost the same.

    A wrong reject loses one idea; a wrong accept spends an implementation trial and then
    sits in the catalog as the thing every later novelty check measures against.
    """
    critic = VotingCritic(verdicts)
    wave = propose_wave(ROOT, name="voted", count=1, rounds=1,
                        client=_one_proposal_client("signed_constraint_parity"),
                        critic_client=critic, critic_samples=3)

    assert len(critic.purposes) == 3
    if accepted:
        assert wave["proposals"][0]["novelty"]["votes"] == votes
        assert wave["proposals"][0]["novelty"]["dissent"] == ["duplicate"]
        assert validate_proposal_wave(wave) == []
    else:
        assert wave["proposals"] == []
        assert wave["rejected"][0]["votes"] == votes
        assert "1/3 samples judged it novel" in wave["rejected"][0]["reason"]


def test_samples_that_skip_a_candidate_leave_the_decision_to_the_ones_that_did_not():
    """Partial failure is the common case with a small model, not an aborted round."""
    critic = VotingCritic([None, "novel", "novel"])
    wave = propose_wave(ROOT, name="partial", count=1, rounds=1,
                        client=_one_proposal_client("signed_constraint_parity"),
                        critic_client=critic, critic_samples=3)

    assert wave["proposals"][0]["novelty"]["votes"] == "2/2"


def test_each_sample_shuffles_the_candidates_and_verdicts_follow_the_proposal():
    """The shuffle is the reason K samples are worth more than one repeated K times.

    Candidates are presented as an ordered list and judged partly against each other, so
    the same order twice mostly reproduces the same opinion twice. Shuffling also means a
    verdict has to be carried back through that sample's own seating: get the mapping
    wrong and the votes land on the neighbouring proposal, which no threshold would catch.
    """
    liked, disliked = "zzq_liked_operation", "zzq_disliked_operation"
    seatings = []

    class SeatAwareCritic:
        model, provider, endpoint = "small", "fake", "http://fake"
        reasoning_effort = None

        def __init__(self):
            self.calls = []

        def json(self, purpose, system, user):
            self.calls.append({"purpose": purpose})
            seating = sorted((liked, disliked), key=user.index)
            seatings.append(seating)
            return {"reviews": [
                {"proposal_id": f"C{seat:03d}",
                 "verdict": "novel" if name == liked else "duplicate",
                 "nearest_neighbors": _neighbors(),
                 "substantive_difference": "a genuinely different operation",
                 "scores": {"novelty": 5 if name == liked else 1, "sft_value": 5,
                            "feasibility": 5, "clarity": 5},
                 "reason": f"judged on {name}"}
                for seat, name in enumerate(seating, 1)]}

    wave = propose_wave(ROOT, name="shuffled", count=2, rounds=1,
                        client=_one_proposal_client(liked, disliked),
                        critic_client=SeatAwareCritic(), critic_samples=6)

    assert [p["name"] for p in wave["proposals"]] == [liked]
    assert {name for name, _ in [tuple(seat) for seat in seatings]} == {liked, disliked}, \
        "every sample presented the same candidate first, so the order was never shuffled"


def test_a_two_hundred_carrying_an_upstream_error_is_retried_not_parsed(monkeypatch):
    """OpenRouter answers 200 and puts the upstream 429 in the body.

    raise_for_status is blind to it, so before this the loop took the failure for an
    answer and the caller died on a KeyError several frames from the cause -- with the
    retries it had earned never spent. Measured against the real gateway, which returned
    `{"error": {"message": "Upstream error from Nvidia: Service temporarily overloaded"}}`
    under an HTTP 200.
    """
    replies = [
        {"error": {"message": "Upstream error from Nvidia: Service temporarily overloaded",
                   "code": 502}},
        {"id": "gen-2", "choices": [{"message": {"content": '{"ok": true}'}}]},
    ]
    posted = []

    class Response:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload
            self.content = json.dumps(payload).encode()

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    def post(url, **kwargs):
        posted.append(url)
        return Response(replies[len(posted) - 1])

    monkeypatch.setattr("reasoning_core.task_search.wave_proposer.requests.post", post)
    monkeypatch.setattr("reasoning_core.task_search.wave_proposer.time.sleep", lambda _: None)

    client = ChatClient(model="m", endpoint="http://x", api_key="k", stream=False)
    assert client.json("probe", "s", "u") == {"ok": True}
    assert len(posted) == 2, "the body-level failure did not cost a retry"
    assert client.calls[0]["response_id"] == "gen-2"


def test_an_upstream_error_that_never_clears_is_raised_not_swallowed(monkeypatch):
    class Response:
        status_code = 200
        content = b'{"error": {"message": "still overloaded"}}'

        def json(self):
            return {"error": {"message": "still overloaded"}}

        def raise_for_status(self):
            return None

    monkeypatch.setattr("reasoning_core.task_search.wave_proposer.requests.post",
                        lambda url, **kwargs: Response())
    monkeypatch.setattr("reasoning_core.task_search.wave_proposer.time.sleep", lambda _: None)

    client = ChatClient(model="m", endpoint="http://x", api_key="k", stream=False)
    with pytest.raises(UpstreamError, match="still overloaded"):
        client.json("probe", "s", "u")


def test_a_critic_batch_is_capped_so_a_truncated_reply_cannot_reject_the_tail():
    """Measured: deepseek-v4-flash returned one review for twenty-four candidates.

    Every review it does not return is scored as a rejection, so an over-large batch does
    not fail loudly -- it silently rejects everything after the truncation point. Twelve
    came back complete, so the batch is capped there and the round is split across calls.
    """
    seen = []

    class Critic:
        model, provider, endpoint = "small", "fake", "http://fake"
        reasoning_effort = None

        def __init__(self):
            self.calls = []

        def json(self, purpose, system, user):
            self.calls.append({"purpose": purpose})
            count = user.count('"proposal_id"') - 1  # the response shape carries one
            seen.append(count)
            return {"reviews": []}

    candidates = [proposal(f"candidate_number_{index:02d}") for index in range(30)]
    critic = Critic()
    _critic_votes(critic, candidates, [], max_catalog_chars=1000, samples=1,
                  round_index=1, wave_name="capped")

    assert max(seen) <= CRITIC_MAX_BATCH, f"sent a batch of {max(seen)} candidates"
    assert sum(seen) == 30, "candidates were dropped rather than split across batches"
    assert [c["purpose"] for c in critic.calls] == [
        "critic-round-1-batch-1", "critic-round-1-batch-2", "critic-round-1-batch-3"]


def test_a_rejected_proposal_keeps_its_summary_so_it_can_be_rejudged():
    """wave9's seventy-one rejected ideas are unrecoverable: the archive kept the reason
    and dropped the summary, and the proposer's replies keep only a sha256. Without the
    summary a rejection cannot be audited, re-judged, or reconsidered later."""
    critic = VotingCritic(["duplicate"])
    wave = propose_wave(ROOT, name="keeps", count=1, rounds=1,
                        client=_one_proposal_client("signed_constraint_parity"),
                        critic_client=critic, critic_samples=1)

    assert wave["rejected"][0]["summary"] == proposal()["summary"]
