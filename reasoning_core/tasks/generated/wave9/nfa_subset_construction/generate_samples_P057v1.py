"""Generate samples_P057v1.md for the nfa_subset_construction task."""

import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.nfa_subset_construction.nfa_subset_construction import (
    NfaSubsetConstruction,
    NfaSubsetConstructionConfig,
)

SEED = 929441144
OUT = Path(__file__).with_name("samples_P057v1.md")


def main():
    random.seed(SEED)
    task = NfaSubsetConstruction()
    task.config = NfaSubsetConstructionConfig()

    blocks = []
    for level in (0, 2, 5):
        blocks.append("## Level {}".format(level))
        blocks.append("")
        task.config.set_level(level)
        for i in range(2):
            e = task.generate_example(level=level)
            blocks.append("### Example {}".format(i + 1))
            blocks.append("")
            blocks.append("**Prompt:**")
            blocks.append("")
            blocks.append("```")
            blocks.append(task.render_prompt(e.metadata))
            blocks.append("```")
            blocks.append("")
            blocks.append("**Answer:**")
            blocks.append("")
            blocks.append("```")
            blocks.append(e.answer)
            blocks.append("```")
            blocks.append("")

    OUT.write_text("\n".join(blocks))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
