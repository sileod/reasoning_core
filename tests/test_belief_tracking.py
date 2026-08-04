import json
from collections import Counter

import pytest

from reasoning_core.template import Entry, edict
from reasoning_core.tasks.belief_tracking import (
    BeliefTracking,
    BeliefTrackingConfig,
    CONTAINER_NAMES,
    EventSpec,
    Percept,
    all_chains,
    _event_from_metadata,
    _event_to_metadata,
    _make_visible,
    _spec_from_metadata,
    _spec_to_metadata,
    full_chain_visibility_interventions,
    backbone_event_necessity,
    certified_report_properties,
    expected_views,
    materialize,
    rematerialized_critical_specs,
    rematerialized_trap_specs,
    replay,
    report_spec_is_sound,
    valid_chains,
)


AGENTS = ("Alice", "Bob", "Carol", "Dave")
INIT = {("loc", "key"): "box"}
FACT = ("loc", "key")


def assert_exact_views(spec, event, depth):
    percept = event.truth if spec.kind == "move" else Percept(
        "claim",
        (spec.actor, spec.asserted_proposition_chain, spec.object, event.details[1], spec.policy),
    )
    assert dict(event.views) == expected_views(spec, percept, AGENTS, depth)


def move(scene="private_observation", observers=("Bob",), destination="drawer", role="critical"):
    return EventSpec(
        kind="move", actor="Alice", scene=scene, object="key", destination=destination,
        observers=observers, role=role,
    )


def report(speaker, listener, source_chain, scene="face_to_face", policy="honest", role="critical"):
    false_claim = "tin" if policy == "deceptive" else None
    return EventSpec(
        kind="report", actor=speaker, target=listener, policy=policy,
        report_type="belief_report", scene=scene,
        object="key", content_source_chain=source_chain,
        asserted_proposition_chain=source_chain, false_claim=false_claim, role=role,
    )


def relay_specs():
    return [
        move(),
        report("Bob", "Carol", ("Bob",)),
        report("Carol", "Alice", ("Carol", "Bob")),
        report("Alice", "Dave", ("Alice", "Carol", "Bob")),
    ]


def test_ordered_chains_allow_nonadjacent_returns():
    assert valid_chains(("Alice", "Bob"), 3) == (
        ("Alice", "Bob", "Alice"),
        ("Bob", "Alice", "Bob"),
    )


def test_grounded_public_private_and_one_way_scenes_derive_views():
    public_spec = move("public_observation", ("Alice", "Bob", "Carol"))
    public = materialize([public_spec], INIT, AGENTS, 3)[0]
    assert_exact_views(public_spec, public, 3)
    assert replay([public], INIT, ("Alice", "Bob", "Alice"))[FACT] == "drawer"

    private_spec = move()
    private = materialize([private_spec], INIT, AGENTS, 2)[0]
    assert_exact_views(private_spec, private, 2)
    assert replay([private], INIT, ("Alice",))[FACT] == "drawer"
    assert replay([private], INIT, ("Bob",))[FACT] == "drawer"
    assert replay([private], INIT, ("Alice", "Bob"))[FACT] == "box"

    one_way_spec = move("one_way_observation", ("Bob", "Carol"))
    one_way = materialize([one_way_spec], INIT, AGENTS, 2)[0]
    assert_exact_views(one_way_spec, one_way, 2)
    assert replay([one_way], INIT, ("Carol", "Bob"))[FACT] == "drawer"
    assert replay([one_way], INIT, ("Bob", "Alice"))[FACT] == "drawer"
    assert replay([one_way], INIT, ("Carol", "Alice"))[FACT] == "drawer"
    assert replay([one_way], INIT, ("Bob", "Carol"))[FACT] == "box"

    one_way_depth_three = materialize([one_way_spec], INIT, AGENTS, 3)[0]
    assert replay([one_way_depth_three], INIT, ("Carol", "Bob", "Alice"))[FACT] == "drawer"


def test_event_views_are_deeply_immutable():
    event = materialize([move()], INIT, AGENTS, 2)[0]
    assert len(event.views) < len(all_chains(AGENTS, 2))
    with pytest.raises(TypeError):
        event.views[("Alice",)] = event.truth


def test_honest_reports_are_rematerialized_from_current_beliefs():
    specs = [move(), report("Bob", "Carol", ("Bob",))]
    trace = materialize(specs, INIT, AGENTS, 2)
    assert_exact_views(specs[1], trace[1], 2)
    assert replay(trace, INIT, ("Carol", "Bob"))[FACT] == "drawer"

    without_observation = materialize(specs[1:], INIT, AGENTS, 2)
    assert replay(without_observation, INIT, ("Carol", "Bob"))[FACT] == "box"


def test_delivery_awareness_depends_on_explicit_mechanism():
    face = report("Bob", "Carol", ("Bob",), scene="face_to_face")
    unconfirmed = report("Bob", "Carol", ("Bob",), scene="unconfirmed_message")
    face_event = materialize([move(), face], INIT, AGENTS, 3)[1]
    unconfirmed_event = materialize([move(), unconfirmed], INIT, AGENTS, 3)[1]
    assert ("Bob", "Carol", "Bob") in face_event.views
    assert ("Bob", "Carol", "Bob") not in unconfirmed_event.views


def test_failed_delivery_has_no_effect():
    specs = [move(), report("Bob", "Carol", ("Bob",), scene="failed_message")]
    trace = materialize(specs, INIT, AGENTS, 2)
    assert_exact_views(specs[1], trace[1], 2)
    assert replay(trace, INIT, ("Carol", "Bob"))[FACT] == "box"


def test_matching_quote_baseline_distinguishes_delivery():
    delivered = report("Bob", "Carol", ("Bob",))
    failed = EventSpec(
        kind="report", actor="Bob", target="Carol", policy="asserted",
        scene="failed_message", object="key", asserted_proposition_chain=("Bob",),
        false_claim="tin",
    )
    specs = [move(), delivered, failed]
    trace = materialize(specs, INIT, AGENTS, 2)
    spans = [
        {"event_index": 1, "delivered": True, "quoted": True,
         "proposition_chain": ["Carol", "Bob"], "object": "key", "container": "drawer"},
        {"event_index": 2, "delivered": False, "quoted": True,
         "proposition_chain": ["Carol", "Bob"], "object": "key", "container": "tin"},
    ]
    baselines = BeliefTracking()._quote_baselines(spans, ("Carol", "Bob"), "key")
    assert baselines["quote:last_delivered_matching"] == "drawer"
    assert baselines["quote:last_matching_including_failed"] == "tin"


def test_container_scoring_accepts_harmless_wrappers_only():
    task = BeliefTracking()
    entry = type("Entry", (), {"answer": "crate"})()

    assert task.score_answer("<crate>", entry) == 1.0
    assert task.score_answer("the crate.", entry) == 1.0
    assert task.score_answer("bowl", entry) == 0.0


def test_outsider_can_see_a_conversation_without_hearing_content():
    visible = EventSpec(
        kind="report", actor="Bob", target="Carol", policy="honest",
        report_type="belief_report",
        scene="visible_conversation", object="key", content_source_chain=("Bob",),
        asserted_proposition_chain=("Bob",), observers=("Dave",), role="distractor",
    )
    trace = materialize([move(), visible], INIT, AGENTS, 2)
    assert_exact_views(visible, trace[1], 2)
    assert replay(trace, INIT, ("Carol", "Bob"))[FACT] == "drawer"
    assert replay(trace, INIT, ("Carol",))[FACT] == "box"
    assert replay(trace, INIT, ("Dave",))[FACT] == "box"


def test_a_speaker_does_not_adopt_their_own_lie():
    lie = report("Bob", "Carol", ("Bob",), policy="deceptive", role="conflict")
    trace = materialize([move(), lie], INIT, AGENTS, 2)
    assert replay(trace, INIT, ("Bob",))[FACT] == "drawer"
    assert replay(trace, INIT, ("Carol", "Bob"))[FACT] == "tin"
    assert replay(trace, INIT, ("Carol",))[FACT] == "box"
    deceptive, conflicts = certified_report_properties([move(), lie], trace, INIT)
    assert deceptive == [1]
    assert conflicts == []


def test_direct_testimony_adoption_is_separate_from_nested_attribution():
    direct = EventSpec(
        kind="report", actor="Bob", target="Carol", policy="honest",
        report_type="direct_claim", scene="face_to_face", object="key",
        content_source_chain=("Bob",), asserted_proposition_chain=(),
        attribution_update=False, adoption_policy="accept",
    )
    trace = materialize([move(), direct], INIT, AGENTS, 2)
    assert_exact_views(direct, trace[1], 2)
    assert replay(trace, INIT, ("Carol",))[FACT] == "drawer"
    assert replay(trace, INIT, ("Carol", "Bob"))[FACT] == "box"

    task = BeliefTracking()
    text = task._event_text(direct, trace[1])
    assert '"The key is in the drawer"' in text
    assert "Carol accepts the stated location" in text


def test_surface_form_is_intent_independent_and_unambiguous():
    hidden_lie = EventSpec(
        kind="report", actor="Bob", target="Carol", policy="deceptive",
        report_type="belief_report", scene="failed_message", object="key",
        content_source_chain=("Bob",), asserted_proposition_chain=("Bob",),
        false_claim="tin", surface_form="indirect_report",
    )
    assert not report_spec_is_sound(hidden_lie)

    indirect = report("Bob", "Carol", ("Bob",))
    indirect = EventSpec(**{**indirect.__dict__, "surface_form": "indirect_report"})
    event = materialize([move(), indirect], INIT, AGENTS, 2)[1]
    text = BeliefTracking()._event_text(indirect, event)
    assert "exactly what Bob believes" in text
    assert " they " not in f" {text} "


def test_reports_state_their_local_semantics_without_global_rules():
    nested = report("Bob", "Carol", ("Bob", "Alice"))
    nested_event = materialize([nested], INIT, AGENTS, 3)[0]
    nested_text = BeliefTracking()._event_text(nested, nested_event)
    assert '"I think Alice thinks the key is in the box"' in nested_text

    unconfirmed = report("Bob", "Carol", ("Bob",), scene="unconfirmed_message")
    unconfirmed_event = materialize([unconfirmed], INIT, AGENTS, 2)[0]
    unconfirmed_text = BeliefTracking()._event_text(unconfirmed, unconfirmed_event)
    assert "Carol receives it, but Bob receives no delivery confirmation" in unconfirmed_text

    direct = EventSpec(
        kind="report", actor="Bob", target="Carol", policy="honest",
        report_type="direct_claim", scene="face_to_face", object="key",
        content_source_chain=("Bob",), asserted_proposition_chain=(),
        attribution_update=False, adoption_policy="accept", surface_form="indirect_report",
    )
    direct_event = materialize([direct], INIT, AGENTS, 2)[0]
    direct_text = BeliefTracking()._event_text(direct, direct_event)
    assert "what Bob believes about where the key is" in direct_text


def test_every_relay_is_causally_critical_after_rematerialization():
    specs = relay_specs()
    chain = ("Dave", "Alice", "Carol", "Bob")
    trace = materialize(specs, INIT, AGENTS, 4)
    assert replay(trace, INIT, chain)[FACT] == "drawer"
    assert rematerialized_critical_specs(specs, INIT, AGENTS, 4, chain, FACT) == [0, 1, 2, 3]
    assert backbone_event_necessity(specs, INIT, AGENTS, 4, chain, FACT, [0, 1, 2, 3]) == [
        0, 1, 2, 3
    ]


def test_repeated_agent_relay_remains_causally_critical():
    specs = [
        move(),
        report("Bob", "Alice", ("Bob",), scene="unconfirmed_message"),
        report("Alice", "Bob", ("Alice", "Bob"), scene="unconfirmed_message"),
    ]
    chain = ("Bob", "Alice", "Bob")
    assert rematerialized_critical_specs(specs, INIT, AGENTS, 3, chain, FACT) == [0, 1, 2]


def test_grounded_visibility_intervention_certifies_a_trap():
    specs = relay_specs()
    specs.insert(1, move(observers=("Dave",), destination="tin", role="trap"))
    chain = ("Dave", "Alice", "Carol", "Bob")
    assert rematerialized_trap_specs(specs, INIT, AGENTS, 4, chain, FACT, [1]) == [1]


def test_awareness_intervention_retains_the_event_and_actual_observer():
    specs = relay_specs() + [move(observers=("Bob",), destination="tin", role="late_news")]
    chain = ("Dave", "Alice", "Carol", "Bob")
    trace = materialize(specs, INIT, AGENTS, 4)
    assert replay(trace, INIT, chain)[FACT] == "drawer"
    assert replay(trace, INIT, ("Bob",))[FACT] == "tin"
    assert full_chain_visibility_interventions(specs, INIT, AGENTS, 4, chain, FACT, [4]) == [4]

    original = specs[4]
    made_visible = _make_visible(original, chain[-1], chain)
    assert made_visible.scene == original.scene
    assert made_visible.observers == original.observers
    assert (chain[-1],) in made_visible.awareness_chains
    assert chain in made_visible.awareness_chains


def test_event_and_spec_serialization_round_trip():
    spec = report("Bob", "Carol", ("Bob",))
    assert _spec_from_metadata(_spec_to_metadata(spec)) == spec
    event = materialize([move()], INIT, AGENTS, 3)[0]
    restored = _event_from_metadata(_event_to_metadata(event))
    assert restored == event
    assert replay([restored], INIT, ("Bob",)) == replay([event], INIT, ("Bob",))


def test_generated_example_has_grounded_certificates_and_no_meta_language():
    task = BeliefTracking(BeliefTrackingConfig(level=3, seed=7))
    entry = task.generate_example()
    task._assert_entry_invariants(entry)
    prompt = entry.prompt.lower()
    assert entry.metadata.query_kind == "belief"
    assert len(entry.metadata.critical_events) == entry.metadata.knobs["critical_event_count"]
    assert entry.metadata.twin_answer != entry.answer
    assert "truthfully" not in prompt and "falsely" not in prompt
    assert prompt.startswith("initially, everyone knows that")
    assert "starting locations are common knowledge" not in prompt
    assert "unseen events and undelivered messages" not in prompt
    assert entry.metadata.requested_target_conflicts == len(
        entry.metadata.certified_target_conflicts
    )
    assert entry.metadata.requested_epistemic_traps <= len(
        entry.metadata.certified_visibility_sensitive_events
    )
    assert not ({"model of", "viewpoint", "registers", "full queried"} & {phrase for phrase in [
        "model of", "viewpoint", "registers", "full queried"
    ] if phrase in prompt})
    for position in entry.metadata.answer_mention_positions:
        assert prompt[position : position + len(entry.answer)] == entry.answer


def test_counterfactuals_are_emitted_as_an_atomic_certified_pair():
    task = BeliefTracking(BeliefTrackingConfig(level=3, hard_fraction=0))
    original, counterfactual = task.generate_examples()
    assert original.answer != counterfactual.answer
    assert not original.metadata.is_counterfactual
    assert counterfactual.metadata.is_counterfactual
    assert original.metadata.twin_answer == counterfactual.answer
    assert counterfactual.metadata.twin_answer == original.answer
    original_key = json.loads(original.deduplication_key)
    counterfactual_key = json.loads(counterfactual.deduplication_key)
    assert original_key.pop("pair_member") is False
    assert counterfactual_key.pop("pair_member") is True
    assert [event["kind"] for event in original_key["proof"]] == [
        event["kind"] for event in counterfactual_key["proof"]
    ]
    task._assert_entry_invariants(original)
    task._assert_entry_invariants(counterfactual)


def test_deduplication_canonicalizes_all_entity_types():
    task = BeliefTracking(BeliefTrackingConfig(level=2, seed=11))
    entry = task.generate_example()
    agent_map = {name: f"Person{i}" for i, name in enumerate(entry.metadata.agents)}
    object_map = {name: f"thing{i}" for i, name in enumerate(entry.metadata.objects)}
    container_map = {name: f"place{i}" for i, name in enumerate(entry.metadata.containers)}
    renaming = {**agent_map, **object_map, **container_map}

    def renamed(value):
        if isinstance(value, dict):
            return {renamed(key): renamed(item) for key, item in value.items()}
        if isinstance(value, list):
            return [renamed(item) for item in value]
        return renaming.get(value, value)

    metadata = edict(renamed(json.loads(json.dumps(entry.metadata))))
    twin = Entry(metadata=metadata, answer=container_map[entry.answer])
    assert task.deduplication_key(entry) == task.deduplication_key(twin)


def test_deduplication_ignores_incidental_events_but_keeps_proof_semantics():
    task = BeliefTracking(BeliefTrackingConfig(level=2, hard_fraction=0))
    entry = task.generate_example()
    incidental = edict(json.loads(json.dumps(entry.metadata)))
    incidental.specs.append(incidental.specs[-1])
    incidental.trace.append(incidental.trace[-1])
    assert task.deduplication_key(entry) == task.deduplication_key(
        Entry(metadata=incidental, answer=entry.answer)
    )

    changed = edict(json.loads(json.dumps(entry.metadata)))
    index = changed.critical_events[0]
    changed.specs[index]["scene"] += "_changed"
    assert task.deduplication_key(entry) != task.deduplication_key(
        Entry(metadata=changed, answer=entry.answer)
    )


def test_shortcut_report_tracks_abstention_coverage():
    entry = Entry(
        metadata=edict(chain=["Alice"], containers=["box", "tin", "drawer"],
                       baselines={"abstaining": None}),
        answer="box",
    )
    report = BeliefTracking().shortcut_report([entry])
    abstaining = report["depth:1:abstaining"]
    assert abstaining["coverage"] == 0
    assert abstaining["conditional_accuracy"] is None
    assert abstaining["advantage"] == 0
    majority = report["depth:1:majority_answer"]
    assert majority["chance"] == 1 / len(CONTAINER_NAMES)


def test_balanced_batch_uses_shared_pairing_balancing_and_deduplication():
    task = BeliefTracking(BeliefTrackingConfig(level=2, hard_fraction=0))
    batch = task.generate_balanced_batch(batch_size=8)
    keys = [json.loads(entry.deduplication_key) for entry in batch]
    assert all(not keys[i]["pair_member"] and keys[i + 1]["pair_member"]
               for i in range(0, len(keys), 2))
    assert all(batch[i].metadata.twin_answer == batch[i + 1].answer
               for i in range(0, len(batch), 2))
    assert len({entry.deduplication_key for entry in batch}) == len(batch)
    assert max(Counter(entry.answer for entry in batch).values()) <= 2
