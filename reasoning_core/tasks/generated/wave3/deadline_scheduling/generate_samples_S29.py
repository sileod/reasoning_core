import os
import random
import textwrap

random.seed(3986056564)

from reasoning_core.tasks.generated.wave3.s29_deadline_scheduling.s29_deadline_scheduling import (
    DeadlineScheduling,
)


def _fmt(e):
    m = e.metadata
    lines = "\n".join(
        f"  Job {i+1}: processing time {t}, deadline {d}"
        for i, (t, d) in enumerate(zip(m.times, m.deadlines))
    )
    prompt = (
        "On a single machine with no preemption, jobs are scheduled one at a time; "
        "a job finishes on time if it completes no later than its deadline. Jobs:\n"
        f"{lines}\n"
        "The answer is the largest number of jobs that can all finish on time."
    )
    return prompt, e.answer


def _render(level):
    task = DeadlineScheduling()
    task.config.set_level(level)
    blocks = []
    for k in range(2):
        prompt, answer = _fmt(task.generate_example())
        blocks.append(f"### Example {k+1}\n\n{prompt}\n\n**Answer:** {answer}")
    return f"Level {level}\n\n" + "\n\n".join(blocks) + "\n"


def main():
    out = "# Deadline scheduling samples\n\n"
    out += _render(0)
    out += "\n" + _render(2)
    out += "\n" + _render(5)
    out = textwrap.dedent(out)
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "samples_S29.md"), "w") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
