import random
from pathlib import Path

from reasoning_core.tasks.generated.wave5.s52_tree_canonical_form.s52_tree_canonical_form import (
    TreeCanonicalForm
)

random.seed(1691368962)


def main():
    task = TreeCanonicalForm()
    task.config.seed = 1691368962
    out = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        out.append(f"## Level {level}")
        out.append("")
        for _ in range(2):
            x = task.generate_example()
            out.append("### Example")
            out.append("")
            out.append(task.render_prompt(x.metadata))
            out.append("")
            out.append("Answer:")
            out.append("")
            out.append(x.answer)
            out.append("")
    path = Path(__file__).with_name("samples_S52.md")
    path.write_text("\n".join(out))
    print("wrote", path)


if __name__ == "__main__":
    main()
