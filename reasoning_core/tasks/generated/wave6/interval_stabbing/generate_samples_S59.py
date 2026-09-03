import random
from pathlib import Path

from reasoning_core.tasks.generated.wave6.s59_interval_stabbing.interval_stabbing import (
    IntervalStabbing,
)

random.seed(348479496)

OUT = Path(__file__).with_name("samples_S59.md")
task = IntervalStabbing()

sections = []
for level in (0, 2, 5):
    rows = []
    for _ in range(2):
        cfg = IntervalStabbing().config_cls
        inst = cfg()
        inst.seed = random.randrange(2 ** 32)
        inst.set_level(level)
        t = IntervalStabbing()
        t.config = inst
        e = t.generate_entry()
        prompt = t.render_prompt(e.metadata)
        rows.append((prompt, e.answer))
    section = "\n\n".join(
        f"**Example {i+1}**\n\nPrompt:\n```\n{p}\n```\n\nAnswer:\n```\n{a}\n```"
        for i, (p, a) in enumerate(rows)
    )
    sections.append(f"## Level {level}\n\n{section}")

md = "\n\n".join(sections) + "\n"
OUT.write_text(md)
print(f"wrote {OUT}")
