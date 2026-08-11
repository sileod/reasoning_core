from reasoning_core.template import Entry, Task


class ConstantLabelTask(Task):
    def __init__(self):
        super().__init__()
        self.index = 0

    def generate_entry(self):
        self.index += 1
        return Entry({"index": self.index}, "True")

    def render_prompt(self, metadata):
        return f"Example {metadata['index']}"

    def score_answer(self, answer, entry):
        return float(str(answer) == entry.answer)


def test_validation_does_not_treat_repeated_labels_as_other_answers():
    task = ConstantLabelTask()
    rows = [Entry({"index": i}, "True") for i in range(4)]
    for row in rows:
        row.prompt = task.render_prompt(row.metadata)

    task._check_validation_examples(rows[0], rows[1:], n_samples=3)
