import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.condorcet_winner.condorcet_winner import (
    CondorcetWinner,
)


def main():
    random.seed(259485130)
    task = CondorcetWinner()
    out = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        out.append(f"# Level {level}\n")
        for _ in range(2):
            e = task.generate_example()
            out.append("## Prompt\n")
            out.append(e.metadata._prompt_text if hasattr(e.metadata, "_prompt_text") else task.render_prompt(e.metadata))
            out.append("\n## Answer\n")
            out.append(e.answer)
            out.append("\n")
    path = Path(__file__).with_name("samples_P063v1.md").resolve()
    # The self-check 'sections' gate compares the level headings and counts
    # 'Answer' under each; write the structured file here.
    path.write_text("\n".join(out))
    print(path)


if __name__ == "__main__":
    main()
