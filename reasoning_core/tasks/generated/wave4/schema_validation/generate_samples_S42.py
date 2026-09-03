import random
import json
from pathlib import Path

random.seed(1755065315)

from reasoning_core.tasks.generated.wave4.s42_schema_validation.s42_schema_validation import SchemaValidation


def main():
    t = SchemaValidation()
    out = []
    for level in (0, 2, 5):
        t.config.set_level(level)
        out.append(f"# Level {level}\n")
        for _ in range(2):
            e = t.generate_example()
            out.append("## Example\n")
            out.append("**Prompt:**\n")
            out.append("```\n" + t.render_prompt(e.metadata) + "\n```\n")
            out.append("**Answer:**\n")
            out.append("```\n" + e.answer + "\n```\n\n")
    return "\n".join(out)


if __name__ == "__main__":
    text = main()
    Path(__file__).with_name("samples_S42.md").write_text(text)
