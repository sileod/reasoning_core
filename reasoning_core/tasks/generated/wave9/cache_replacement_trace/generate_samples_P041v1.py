"""Generate samples_P041v1.md for the cache_replacement_trace task."""

import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.cache_replacement_trace.cache_replacement_trace import (
    CacheReplacementTrace,
)

SEED = 1492106328
OUT = Path(__file__).with_name("samples_P041v1.md")


def main():
    random.seed(SEED)
    out = []
    out.append("# Sample cache_replacement_trace (P041v1)")
    out.append("")
    out.append("Task: execute LRU, LFU, or FIFO cache accesses with inserts and evictions "
               "under explicit tie rules, returning hits, misses, or final cache state.")
    out.append("")
    task = CacheReplacementTrace()
    for level in (0, 2, 5):
        task.config.set_level(level)
        out.append(f"## Level {level}")
        out.append("")
        for _ in range(2):
            e = task.generate_example()
            out.append("**Prompt:**")
            out.append("")
            out.append(task.render_prompt(e.metadata))
            out.append("")
            out.append("**Answer:**")
            out.append("")
            out.append(e.answer)
            out.append("")
    with open(OUT, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
