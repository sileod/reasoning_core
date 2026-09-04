import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from reasoning_core.tasks.generated.wave9.quorum_read_write_resolution.quorum_read_write_resolution import QuorumReadWriteResolution

OUT = Path(__file__).with_name("samples_P079v1.md")


def main():
    random.seed(1175081579)
    task = QuorumReadWriteResolution()
    with open(OUT, "w") as f:
        for level in (0, 2, 5):
            task.config.set_level(level)
            f.write("## Level %d\n\n" % level)
            for _ in range(2):
                ex = task.generate_example()
                f.write("### Example\n\n")
                f.write("**Prompt:**\n\n")
                f.write(task.render_prompt(ex.metadata) + "\n\n")
                f.write("**Answer:**\n\n")
                f.write(ex.answer + "\n\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
