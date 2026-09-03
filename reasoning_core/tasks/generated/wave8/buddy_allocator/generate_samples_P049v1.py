import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.buddy_allocator.buddy_allocator import BuddyAllocator

SEED = 545695329


def main():
    random.seed(SEED)
    task = BuddyAllocator()
    out = []
    for level in (0, 2, 5):
        out.append(f"## Level {level}\n")
        for k in range(2):
            task.config.set_level(level)
            entry = task.generate_example()
            prompt = task.render_prompt(entry.metadata)
            out.append(f"### Example {k+1}\n")
            out.append(prompt)
            out.append("\n\n**Answer:**")
            out.append(entry.answer)
            out.append("\n")
        out.append("\n")
    path = Path(__file__).with_name("samples_P049v1.md")
    path.write_text("\n".join(out))


if __name__ == "__main__":
    main()
