import random
from pathlib import Path

random.seed(1787056888)

from reasoning_core.tasks.generated.wave9.minimal_unsat_core.minimal_unsat_core import MinimalUnsatCore


def render(task, entry):
    return task.render_prompt(entry.metadata)


def main():
    out = []
    for level in (0, 2, 5):
        out.append(f"# Level {level}")
        for _ in range(2):
            task = MinimalUnsatCore()
            task.config.seed = 1787056888
            entry = task.generate_example(level=level)
            out.append("**Prompt:**")
            out.append(render(task, entry))
            out.append("**Answer:**")
            out.append(entry.answer)
            out.append("")
    text = "\n".join(out)
    dest = Path(__file__).with_name("samples_P051v1.md")
    dest.write_text(text)
    print("wrote", dest)


if __name__ == "__main__":
    main()
