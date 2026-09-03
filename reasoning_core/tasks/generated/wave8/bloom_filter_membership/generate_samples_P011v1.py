import random
from pathlib import Path

import importlib

mod = importlib.import_module(
    "reasoning_core.tasks.generated.wave8.bloom_filter_membership.bloom_filter_membership"
)


def main():
    random.seed(244270231)
    out = []
    for level in [0, 2, 5]:
        out.append(f"# Level {level}")
        t = mod.BloomFilterMembership()
        t.config.set_level(level)
        for _ in range(2):
            e = t.generate_example()
            prompt = t.render_prompt(e.metadata)
            out.append("## Example")
            out.append("Prompt:")
            out.append("```")
            out.append(prompt)
            out.append("```")
            out.append("Answer:")
            out.append("```")
            out.append(e.answer)
            out.append("```")
    text = "\n".join(out) + "\n"
    dest = Path(__file__).with_name("samples_P011v1.md")
    dest.write_text(text)
    print(dest)


if __name__ == "__main__":
    main()
