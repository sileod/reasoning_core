import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from reasoning_core.template import Config

from reasoning_core.tasks.generated.wave8.bcnf_violation.bcnf_violation import (
    BcnfViolation, _closure, _is_superkey, _first_violation,
)


def _check_gold(entry, task):
    assert task.score_answer(entry.answer, entry) == 1.0


def test_generate_and_score_levels():
    random.seed(4052403849)
    task = BcnfViolation()
    for level in range(7):
        cfg = BcnfViolation.config_cls()
        cfg.set_level(level)
        task.config = cfg
        seen = set()
        for _ in range(20):
            entry = task.generate_entry()
            _check_gold(entry, task)
            seen.add(entry.answer)
        assert len(seen) > 1


def test_junk_and_empty_score_zero():
    random.seed(123)
    task = BcnfViolation()
    entry = task.generate_entry()
    for junk in ("", "A", "random text", "{} -> B extra"):
        assert task.score_answer(junk, entry) < 1.0


def test_answer_distribution_both_labels():
    random.seed(999)
    task = BcnfViolation()
    none_count = 0
    viol_count = 0
    for _ in range(200):
        entry = task.generate_entry()
        if entry.answer == "None":
            none_count += 1
        else:
            viol_count += 1
    assert viol_count > 20
    assert none_count > 5
