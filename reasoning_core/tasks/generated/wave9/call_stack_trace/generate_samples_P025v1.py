import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.call_stack_trace.call_stack_trace import (
    CallStackTrace, CallStackConfig)

SEED = 3124932900


def main():
    random.seed(SEED)
    out_path = Path(__file__).with_name("samples_P025v1.md")
    lines = []
    for level in (0, 2, 5):
        cfg = CallStackConfig()
        cfg.set_level(level)
        t = CallStackTrace(config=cfg)
        lines.append("## Level %d" % level)
        for i in range(2):
            e = t.generate_example()
            prompt = t.render_prompt(e.metadata)
            lines.append("### Example %d" % (i + 1))
            lines.append("**Prompt:**")
            lines.append(prompt)
            lines.append("")
            lines.append("**Answer:**")
            lines.append(e.answer)
            lines.append("")
    out_path.write_text("\n".join(lines) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
