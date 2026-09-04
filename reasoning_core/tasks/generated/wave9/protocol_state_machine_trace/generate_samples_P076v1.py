"""Generate samples_P076v1.md reproducibly.

Run once in a fresh process with a fixed seed so the bytes are stable.
"""

import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.protocol_state_machine_trace.protocol_state_machine_trace import (
    ProtocolStateMachineTrace,
)

SEED = 4137643316
OUT = Path(__file__).with_name("samples_P076v1.md")


def main():
    lines = []
    for level in (0, 2, 5):
        lines.append("# Level %d\n" % level)
        random.seed(SEED + level)
        task = ProtocolStateMachineTrace()
        for _ in range(2):
            x = task.generate_example()
            lines.append("## Example\n")
            lines.append("Prompt:\n")
            lines.append("```\n%s\n```\n" % task.render_prompt(x.metadata))
            lines.append("Answer: %s\n" % x.answer)
        lines.append("\n")
    OUT.write_text("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
