from reasoning_core.template import Entry, Task


class CandidateTask(Task):
    def generate_entry(self):
        return Entry({}, "unused")

    def render_prompt(self, metadata):
        return "prompt"

    def distractor_candidates(self, entry):
        yield from ("10", "11", "11", "12", None, "bad")

    def score_answer(self, answer, entry):
        if answer == "bad":
            raise ValueError("malformed")
        return 1 if answer in {entry.answer, "12"} else 0.5


class FallbackTask(Task):
    def __init__(self, answers):
        self.answers = iter(answers)
        self.generation_count = 0
        super().__init__()

    def generate_entry(self):
        self.generation_count += 1
        return Entry({}, next(self.answers))

    def render_prompt(self, metadata):
        return "prompt"


class ConstantTask(Task):
    def __init__(self):
        self.generation_count = 0
        super().__init__()

    def generate_entry(self):
        self.generation_count += 1
        return Entry({}, "yes")

    def render_prompt(self, metadata):
        return "prompt"


class RankingTask(CandidateTask):
    def distractor_candidates(self, entry):
        yield from ("completely different", "gold!")


def test_task_candidates_are_validated_deduplicated_and_ranked():
    task = CandidateTask()
    entry = Entry({}, "10")

    assert task.generate_distractors(entry, n=3, max_candidates=6) == ["11"]


def test_ranking_can_promote_a_later_surface_match():
    task = RankingTask()

    assert task.generate_distractors(Entry({}, "gold"), n=1, max_candidates=2) == ["gold!"]


def test_normal_generation_populates_bounded_unique_answer_reservoir():
    task = FallbackTask(str(i) for i in range(70))
    for _ in range(70):
        task.generate_example()

    assert task._answer_reservoir == [str(i) for i in range(6, 70)]


def test_fallback_reuses_observed_answers_without_generating_when_sufficient():
    task = FallbackTask(iter(()))
    task._answer_reservoir = ["near", "far"]
    entry = Entry({}, "gold")

    assert task.generate_distractors(entry, n=2) == ["near", "far"]
    assert task.generation_count == 0


def test_fallback_generates_same_task_entries_only_as_needed():
    task = FallbackTask(["gold", "wrong-1", "wrong-2"])
    entry = Entry({}, "gold")

    distractors = task.generate_distractors(entry, n=2, max_candidates=3)

    assert set(distractors) == {"wrong-1", "wrong-2"}
    assert task.generation_count == 3


def test_low_cardinality_fallback_stops_after_vocabulary_saturates():
    task = ConstantTask()

    assert task.generate_distractors(Entry({}, "yes"), max_candidates=64) == []
    assert task.generation_count == task._distractor_saturation_patience + 1


def test_candidate_budget_and_zero_limits_are_hard_bounds():
    task = FallbackTask(["a", "b", "c"])
    entry = Entry({}, "gold")

    assert len(task.generate_distractors(entry, n=16, max_candidates=2)) == 2
    assert task.generation_count == 2
    assert task.generate_distractors(entry, n=0) == []
    assert task.generate_distractors(entry, max_candidates=0) == []


def test_candidate_iterator_is_not_advanced_past_budget():
    yielded = []

    class CountingTask(CandidateTask):
        def distractor_candidates(self, entry):
            for i in range(10):
                yielded.append(i)
                yield str(i)

    CountingTask().generate_distractors(Entry({}, "gold"), max_candidates=3)

    assert yielded == [0, 1, 2]
