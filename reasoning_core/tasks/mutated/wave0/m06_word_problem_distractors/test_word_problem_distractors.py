import random
import sys
sys.path.insert(0, ".")

from reasoning_core.tasks.mutated.wave0.m06_word_problem_distractors.word_problem_distractors import (
    WordProblemDistractors,
    WordProblemDistractorConfig,
    proof_core_size,
)


def test_gold_scores_one():
    for level in (0, 2, 5):
        task = WordProblemDistractors()
        task.config.set_level(level)
        problem = task.generate_entry()
        assert task.score_answer(problem.answer, problem) == 1.0


def test_has_distractors():
    task = WordProblemDistractors()
    problem = task.generate_entry()
    m = problem.metadata
    assert m.distractor_count >= 1
    assert len(m.distractor_names) == m.distractor_count
    assert all(n in m.names for n in m.distractor_names)


def test_irrelevance_formally_holds():
    task = WordProblemDistractors()
    for _ in range(50):
        problem = task.generate_entry()
        m = problem.metadata
        core = proof_core_size(m.names, m.relations, m.given, m.asked, m.given_value)
        assert core == m.proof_core_size


def test_wrong_answer_scores_low():
    task = WordProblemDistractors()
    for _ in range(20):
        problem = task.generate_entry()
        correct = int(problem.answer)
        wrong = correct + 5
        assert task.score_answer(str(wrong), problem) < 1.0


def test_difficulty_increases():
    low = WordProblemDistractorConfig()
    low.set_level(0)
    high = WordProblemDistractorConfig()
    high.set_level(5)
    assert high.n_rel >= low.n_rel
    assert high.max_n > low.max_n
    assert high.chain_max >= low.chain_max


def test_provenance_meta():
    import reasoning_core.tasks.mutated.wave0.m06_word_problem_distractors.word_problem_distractors as mod
    assert mod.TASK_META["parent_source_id"] == (
        "c267a83e5953e4862bec61fb7c72a249dc6d8d945f1116585ac947e52ef26f35"
    )
    assert mod.TASK_META["hypothesis"] == "H2"


def test_seed_reproducible():
    random.seed(2364728918)
    task = WordProblemDistractors()
    task.config.set_level(2)
    first = task.generate_entry()
    random.seed(2364728918)
    task2 = WordProblemDistractors()
    task2.config.set_level(2)
    second = task2.generate_entry()
    assert first.answer == second.answer
    assert list(first.metadata.relations) == list(second.metadata.relations)
