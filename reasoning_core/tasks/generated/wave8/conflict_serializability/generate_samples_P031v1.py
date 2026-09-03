import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.conflict_serializability.conflict_serializability import (
    ConflictSerializability,
    ConflictSerializabilityConfig,
)

SEED = 2300412284


def _emit(out, task, levels):
    for level in levels:
        cfg = ConflictSerializabilityConfig()
        cfg.set_level(level)
        t = ConflictSerializability(config=cfg)
        out.append(f"## Level {level}\n")
        for k in range(2):
            ex = t.generate_example()
            out.append(f"### Example {k+1}\n")
            out.append("Prompt:")
            out.append("```")
            out.append(t.render_prompt(ex.metadata))
            out.append("```")
            out.append("Answer:")
            out.append("```")
            out.append(ex.answer)
            out.append("```")
            out.append("")
    return out


def main():
    random.seed(SEED)
    out = ["# Samples P031v1", ""]
    outer = ConflictSerializability()
    _emit(out, outer, (0, 2, 5))
    path = Path(__file__).with_name("samples_P031v1.md")
    path.write_text("\n".join(out))


if __name__ == "__main__":
    main()
