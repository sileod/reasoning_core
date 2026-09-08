"""Generate samples_P002v1.md for the request_revision task."""

import random
from pathlib import Path

from reasoning_core.tasks.generated.wave10.request_revision.request_revision import (
    RequestRevision,
)


def main():
    random.seed(3926271670)
    out = ["# samples_P002v1", ""]
    task = RequestRevision()
    for level in (0, 2, 5):
        out.append(f"## Level {level}")
        for _ in range(2):
            e = task.generate_example(level=level)
            prompt = task.render_prompt(e.metadata)
            out.append("**Prompt:**")
            out.append("```")
            out.append(prompt)
            out.append("```")
            out.append("**Answer:**")
            out.append("```")
            out.append(e.answer)
            out.append("```")
            out.append("")
    Path(__file__).with_name("samples_P002v1.md").write_text("\n".join(out))


if __name__ == "__main__":
    main()
