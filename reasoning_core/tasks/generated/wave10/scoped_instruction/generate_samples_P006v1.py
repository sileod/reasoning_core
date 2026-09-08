import random
from pathlib import Path

from reasoning_core.tasks.generated.wave10.scoped_instruction.scoped_instruction import (
    ScopedInstruction,
)


def _render(level, prompt, answer):
    return f"**Level {level}**\n\nPrompt:\n{prompt}\n\nAnswer:\n{answer}\n"


def main():
    random.seed(4293640035)
    seed_calls = 0
    t = ScopedInstruction()
    out = []
    for level in (0, 2, 5):
        cfg = type(t.config)()
        cfg.set_level(level)
        for _ in range(2):
            t.config = cfg
            e = t.generate_entry()
            prompt = t.render_prompt(e.metadata)
            out.append(_render(level, prompt, e.answer))

    text = "\n".join(out) + "\n"
    path = Path(__file__).with_name("samples_P006v1.md")
    path.write_text(text)


if __name__ == "__main__":
    main()
