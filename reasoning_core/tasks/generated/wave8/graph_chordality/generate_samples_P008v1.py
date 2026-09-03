import random
from pathlib import Path
from reasoning_core.tasks.generated.wave8.graph_chordality.graph_chordality import (
    GraphChordality,
)

SEED = 1705491944
OUT = Path(__file__).with_name("samples_P008v1.md")


def main():
    out = []
    for level in (0, 2, 5):
        random.seed(SEED + level)
        out.append(f"# Level {level}")
        for i in range(2):
            t = GraphChordality()
            t.config.set_level(level)
            ex = t.generate_example()
            out.append(f"## Example {i + 1}")
            out.append("**Prompt:**")
            out.append("")
            out.append(t.render_prompt(ex.metadata))
            out.append("")
            out.append(f"**Answer:** `{ex.answer}`")
            out.append("")
    OUT.write_text("\n".join(out) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
