from reasoning_core.tasks.qstr import (
    QualitativeReasoning,
    QualitativeReasoningConfig,
    _rank_candidates,
    rcc8_query_relations,
)


def test_ordinal_ranking_generation_is_mixed_and_unique():
    task = QualitativeReasoning(
        QualitativeReasoningConfig(n_entities=5, ordinal_prob=1.0)
    )

    seen_kinds = set()
    for _ in range(20):
        problem = task.generate_example(max_tokens=0)
        kinds = {clue["kind"] for clue in problem.metadata.clues}
        seen_kinds.update(kinds)

        assert problem.metadata.family == "ordinal"
        assert len(kinds) >= 2
        assert all(
            clue.get("rank") != problem.metadata.query_rank
            for clue in problem.metadata.clues
            if clue["kind"] == "rank"
        )
        assert _rank_candidates(
            problem.metadata.n_entities,
            problem.metadata.clues,
            problem.metadata.query_rank,
        ) == [problem.answer]
        assert task.score_answer(problem.answer, problem) == 1.0
        assert task.score_answer("not-an-entity", problem) == 0.0
    assert seen_kinds == {"pair", "rank", "next"}


def test_original_qualitative_generation_remains_available():
    task = QualitativeReasoning(QualitativeReasoningConfig(ordinal_prob=0.0))
    problem = task.generate_example(max_tokens=0)

    assert problem.metadata.get("family") != "ordinal"
    assert problem.metadata.calculus
    assert task.score_answer(problem.answer, problem) == 1.0


def test_rcc8_box_composition_includes_2d_cases():
    task = QualitativeReasoning(QualitativeReasoningConfig(ordinal_prob=0.0))
    calc = task.registry['rcc8']['calc']
    assert calc.compose['EC', 'EC'] == frozenset(
        {'DC', 'EC', 'PO', 'EQ', 'TPP', 'TPPi'}
    )


def test_rcc8_box_composition_is_cached_across_tasks():
    first = QualitativeReasoning().registry['rcc8']['calc']
    second = QualitativeReasoning().registry['rcc8']['calc']
    assert first is second


def test_rcc8_query_relations_detects_ambiguity():
    candidates = rcc8_query_relations(
        3,
        [(0, 1, 'EC'), (1, 2, 'EC')],
        (0, 2),
    )
    assert candidates == frozenset(
        {'DC', 'EC', 'PO', 'EQ', 'TPP', 'TPPi'}
    )
