import random

from reasoning_core.tasks.generated.wave8.schedule_recoverability import \
    schedule_recoverability as mod
from reasoning_core.tasks.generated.wave8.schedule_recoverability.\
    schedule_recoverability import _classify, _make
from reasoning_core.template import Task


def test_import_and_meta():
    t = mod.ScheduleRecoverability()
    assert isinstance(mod.TASK_META, dict)
    assert mod.TASK_META["idea"] == "schedule_recoverability (draw 2 of 2)"


def test_summary_one_line():
    s = mod.ScheduleRecoverability.summary
    assert isinstance(s, str) and "\n" not in s and s == s.strip()


def test_generate_scores_one():
    t = mod.ScheduleRecoverability()
    random.seed(0)
    counts = {}
    for _ in range(40):
        ex = t.generate_example()
        assert t.score_answer(ex.answer, ex) == 1.0
        counts[ex.answer] = counts.get(ex.answer, 0) + 1
    assert max(counts.values()) / 40 < 0.5


def test_exhaustive_classification():
    rng = random.Random(123)
    counts = {}
    for _ in range(2000):
        ops = _make(rng.choice(["unrecoverable", "recoverable", "cascadeless", "strict"]),
                    3, 3, 2)
        cls, _ = _classify(ops)
        counts[cls] = counts.get(cls, 0) + 1
    assert set(counts) <= {"strict", "cascadeless", "recoverable", "unrecoverable"}


def test_class_always_realizes_target():
    for target in ["unrecoverable", "recoverable", "cascadeless", "strict"]:
        for _ in range(300):
            ops = _make(target, 4, 4, 3)
            assert _classify(ops)[0] == target, (target, ops)


def test_score_rejects_garbage_and_empty():
    t = mod.ScheduleRecoverability()
    ex = t.generate_example()
    assert t.score_answer("", ex) < 1
    assert t.score_answer("reajrjrje9595!", ex) < 1
    assert t.score_answer("import fakemodule", ex) < 1


def test_difficulty_changes():
    t = mod.ScheduleRecoverability()
    c0 = t.config.to_dict()
    t.config.set_level(3)
    assert t.config.to_dict() != c0


def test_all_levels_generate():
    t = mod.ScheduleRecoverability()
    for level in range(0, 7):
        ex = t.generate_example(level=level)
        assert t.score_answer(ex.answer, ex) == 1.0


def test_labels_and_witness_vary():
    t = mod.ScheduleRecoverability()
    random.seed(7)
    answers = set(t.generate_example().answer for _ in range(40))
    labels = {a.split()[0] for a in answers}
    assert labels & {"unrecoverable", "recoverable", "cascadeless", "strict"}
    wit = {int(a.split()[1]) for a in answers}
    assert len(wit) > 2
