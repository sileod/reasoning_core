from pathlib import Path

import random

from reasoning_core.tasks.generated.wave9.bipartite_matching.bipartite_matching import BipartiteMatching


def main():
    random.seed(3986913703)
    task = BipartiteMatching()
    out = ["# Samples for P015v1 (bipartite_matching)\n"]
    for level in (0, 2, 5):
        out.append("## Level %d\n" % level)
        for i in range(2):
            ex = task.generate_example(level=level)
            out.append("### Example %d\n" % (i + 1))
            out.append("Prompt:\n%s\n" % ex.prompt)
            out.append("Answer:\n%s\n" % ex.answer)
    path = Path(__file__).with_name("samples_P015v1.md")
    path.write_text("\n".join(out), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
