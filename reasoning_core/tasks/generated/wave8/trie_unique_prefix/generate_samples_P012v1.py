import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.trie_unique_prefix.trie_unique_prefix import TrieUniquePrefix


def main():
    random.seed(1967867643)
    task = TrieUniquePrefix()
    out = []
    for level in (0, 2, 5):
        task.config.set_level(level)
        out.append(f"## Level {level}")
        for _ in range(2):
            ex = task.generate_example()
            prompt = task.render_prompt(ex.metadata)
            out.append("### Prompt")
            out.append(prompt)
            out.append("")
            out.append("### Answer")
            out.append(ex.answer)
            out.append("")
    path = Path(__file__).with_name("samples_P012v1.md")
    path.write_text("\n".join(out))
    print(path.resolve())


if __name__ == "__main__":
    main()
