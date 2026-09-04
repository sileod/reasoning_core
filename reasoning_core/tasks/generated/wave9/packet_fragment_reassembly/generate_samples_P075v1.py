import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.packet_fragment_reassembly import packet_fragment_reassembly as mod

SEED = 2968879882


def main():
    random.seed(SEED)
    out = Path(__file__).with_name("samples_P075v1.md")
    task = mod.PacketFragmentReassembly()
    lines = []
    for level in (0, 2, 5):
        lines.append(f"## Level {level}")
        lines.append("")
        task.config.set_level(level)
        for i in range(2):
            e = task.generate_entry()
            prompt = task.render_prompt(e.metadata)
            lines.append(f"### Example {i + 1}")
            lines.append("")
            lines.append("Prompt:")
            lines.append("")
            lines.append("```")
            lines.append(prompt)
            lines.append("```")
            lines.append("")
            lines.append("Answer:")
            lines.append("")
            lines.append(f"```\n{e.answer}\n```")
            lines.append("")
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
