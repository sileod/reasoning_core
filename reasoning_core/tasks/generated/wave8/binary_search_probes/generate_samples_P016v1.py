import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.binary_search_probes.binary_search_probes import (
    BinarySearchProbes,
    _binary_search_probes,
)


def main():
    random.seed(3549874681)
    task = BinarySearchProbes()
    out = Path(__file__).with_name("samples_P016v1.md")
    lines = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        lines.append(f"## Level {level}\n")
        for _ in range(2):
            entry = task.generate_example()
            prompt = task.render_prompt(entry.metadata)
            arr = entry.metadata.arr
            target = entry.metadata.target
            gold = _binary_search_probes(arr, target)
            lines.append(f"**Prompt:** {prompt}\n")
            lines.append(f"**Answer:** {entry.answer}  (gold probes: {gold})\n")
        lines.append("")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
