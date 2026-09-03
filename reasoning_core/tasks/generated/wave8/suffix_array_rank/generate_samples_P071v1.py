import random
from pathlib import Path

random.seed(3317076631)

from reasoning_core.tasks.generated.wave8.suffix_array_rank.suffix_array_rank import SuffixRank


def main():
    task = SuffixRank()
    out = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        out.append(f"## Level {level}\n")
        for _ in range(2):
            entry = task.generate_example()
            prompt = task.render_prompt(entry.metadata)
            out.append(f"### Prompt\n\n{prompt}\n")
            out.append(f"### Answer\n\n{entry.answer}\n")

    path = Path(__file__).with_name("samples_P071v1.md")
    path.write_text("\n".join(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
