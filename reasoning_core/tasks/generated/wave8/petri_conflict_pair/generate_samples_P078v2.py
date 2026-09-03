import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.petri_conflict_pair.petri_conflict_pair import (
    PetriConflictPair,
)

SEED = 1253355878


def main():
    random.seed(SEED)
    task = PetriConflictPair()
    out_path = Path(__file__).with_name("samples_P078v2.md")
    lines = []
    for level in (0, 2, 5):
        lines.append('## Level %d' % level)
        lines.append('')
        for _ in range(2):
            e = task.generate_example(level=level)
            lines.append(task.render_prompt(e.metadata))
            lines.append('')
            lines.append('Answer: %s' % e.answer)
            lines.append('')
    out_path.write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
