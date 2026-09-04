import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')))

from reasoning_core.tasks.generated.wave9.temporal_logic_monitoring.temporal_logic_monitoring import (  # noqa: E501
    TemporalLogicMonitoring,
)

SEED = 43127990
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples_P056v1.md')


def main():
    random.seed(SEED)
    task = TemporalLogicMonitoring()
    levels = [0, 2, 5]
    lines = []
    for level in levels:
        lines.append('# Level %d' % level)
        lines.append('')
        task.config.set_level(level)
        for i in range(2):
            ex = task.generate_example()
            lines.append('## Example %d (level %d)' % (i + 1, level))
            lines.append('')
            lines.append('**Prompt**')
            lines.append('```')
            lines.append(task.render_prompt(ex.metadata))
            lines.append('```')
            lines.append('')
            lines.append('**Answer**')
            lines.append('```')
            lines.append(ex.answer)
            lines.append('```')
            lines.append('')
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
