import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from reasoning_core.tasks.generated.wave9.ll1_predictive_parsing.ll1_predictive_parsing import (
    LL1PredictiveParsing, LL1ParsingConfig, _trace,
)


def _parse_answer(answer):
    return answer


def test_generate_and_score():
    random.seed(1742980165)
    task = LL1PredictiveParsing()
    task.config.set_level(2)
    for _ in range(50):
        entry = task.generate_example()
        assert task.score_answer(entry.answer, entry) == 1.0
        assert task.score_answer('', entry) < 1.0
        assert task.score_answer('garbage', entry) < 1.0


def test_trace_accepts_and_is_deterministic():
    random.seed(1742980165)
    task = LL1PredictiveParsing()
    for level in (0, 3, 6):
        task.config.set_level(level)
        for _ in range(20):
            entry = task.generate_example()
            table = {(r[0], r[1]): r[2] for r in entry.metadata.table}
            trace, outcome = _trace(table, entry.metadata.start,
                                    entry.metadata.input.split())
            assert outcome == 'accept'
            assert ' accept' in entry.answer


def test_difficulty_changes_config():
    task = LL1PredictiveParsing()
    task.config.set_level(0)
    c0 = (task.config.n_prod, task.config.max_apply)
    task.config.set_level(6)
    c6 = (task.config.n_prod, task.config.max_apply)
    assert c6 != c0
