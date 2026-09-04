import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from reasoning_core.tasks.generated.wave9.ll1_predictive_parsing.ll1_predictive_parsing import (
    LL1PredictiveParsing,
)

SEED = 1742980165


def main():
    random.seed(SEED)
    out = []
    task = LL1PredictiveParsing()
    for level in (0, 2, 5):
        task.config.set_level(level)
        out.append('# Level %d' % level)
        for _ in range(2):
            entry = task.generate_example()
            out.append('## Example')
            out.append('**Prompt:**')
            out.append('```')
            out.append(task.render_prompt(entry.metadata))
            out.append('```')
            out.append('**Answer:**')
            out.append('```')
            out.append(entry.answer)
            out.append('```')
    path = Path(__file__).with_name('samples_P060v1.md')
    path.write_text('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
