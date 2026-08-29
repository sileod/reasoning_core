import random
import pathlib

from reasoning_core.tasks.mutated.wave0.m05_internal_process_inversion.process_state import ProcessState as T

random.seed(2033032770)

out = []
for level in (0, 2, 5):
    task = T()
    task.config.set_level(level)
    out.append(f"## Level {level}\n")
    for _ in range(2):
        ex = task.generate_example()
        out.append("### Prompt\n")
        out.append(task.render_prompt(ex.metadata))
        out.append("\n")
        out.append(f"### Answer\n\n{ex.answer}\n")
        out.append("\n")

path = pathlib.Path(__file__).parent / "samples_M5.md"
path.write_text("\n".join(out))
print(path)
