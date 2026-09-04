import random

from reasoning_core.tasks.generated.wave0.n01_congruence_closure.n01_congruence_closure import CongruenceClosure


random.seed(1527083350)

levels = [0, 1, 2, 3, 4, 5]
out = []
for level in levels:
    task = CongruenceClosure()
    cfg = task.config
    cfg.set_level(level)
    out.append("## Level %d" % level)
    for _ in range(3):
        ex = task.generate_example()
        out.append("### Prompt")
        out.append("```")
        out.append(task.render_prompt(ex.metadata))
        out.append("```")
        out.append("### Answer")
        out.append("```")
        out.append(ex.answer)
        out.append("```")

with open("reasoning_core/tasks/generated/wave0/n01_congruence_closure/samples_N1.md", "w") as f:
    f.write("\n".join(out) + "\n")
