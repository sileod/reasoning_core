import random
from pathlib import Path

random.seed(3326285053)

from reasoning_core.tasks.generated.wave9.rate_limit_token_bucket.rate_limit_token_bucket import (
    RateLimitTokenBucket,
)

OUT = Path(__file__).with_name("samples_P045v1.md")
TASK = RateLimitTokenBucket()

levels = [0, 2, 5]

lines = []
lines.append("# Samples P045v1\n")
for level in levels:
    for i in range(2):
        entry = TASK.generate_example(level=level)
        prompt = TASK.render_prompt(entry.metadata)
        lines.append(f"## Level {level}")
        lines.append(f"### Example {i + 1}")
        lines.append("**Prompt:**")
        lines.append("")
        lines.append(prompt)
        lines.append("")
        lines.append("**Answer:**")
        lines.append("")
        lines.append(entry.answer)
        lines.append("")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT}")
