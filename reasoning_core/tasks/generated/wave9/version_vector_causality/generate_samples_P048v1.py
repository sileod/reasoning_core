import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.version_vector_causality.version_vector_causality import (
    VersionVectorCausality, VectorCausalityConfig)

SEED = 2951849487


def main():
    random.seed(SEED)
    out = []
    for level in (0, 2, 5):
        cfg = VectorCausalityConfig()
        cfg.set_level(level)
        t = VersionVectorCausality(config=cfg)
        out.append(f"# Level {level}\n")
        for i in range(2):
            e = t.generate_entry()
            prompt = t.render_prompt(e.metadata)
            out.append(f"## Example {i+1}\n")
            out.append("**Prompt:**\n")
            out.append(f"```\n{prompt}\n```\n")
            out.append("**Answer:**\n")
            out.append(f"```\n{e.answer}\n```\n")
        out.append("\n")
    path = Path(__file__).with_name("samples_P048v1.md")
    path.write_text("\n".join(out))


if __name__ == "__main__":
    main()
