"""Generate samples_P003v1.md for the bridge_edges task."""

import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.bridge_edges.bridge_edges import BridgeEdgesTask


def main():
    random.seed(2592893641)
    task = BridgeEdgesTask()
    out = Path(__file__).with_name("samples_P003v1.md")
    parts = ["# bridge_edges samples", ""]
    parts.append(
        "The answer format is the semicolon-separated list of bridge edges, "
        "each edge as the two node numbers with the smaller first (e.g. 1-4), "
        "listed in lexicographic order, or 'none' if the graph has no bridges."
    )
    for level in (0, 2, 5):
        parts.append(f"")
        parts.append(f"## Level {level}")
        parts.append("")
        for _ in range(2):
            task.config.set_level(level)
            entry = task.generate_example()
            parts.append("Prompt:")
            parts.append("")
            parts.append(task.render_prompt(entry.metadata))
            parts.append("")
            parts.append("Answer:")
            parts.append("")
            parts.append(entry.answer)
            parts.append("")
    out.write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
