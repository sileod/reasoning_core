import random
from reasoning_core.tasks.generated.wave0.n11_interval_abstract_interpretation.interval_ai import IntervalAI

random.seed(2110083903)

out = []
for level in (0, 2, 5):
    out.append(f"# Level {level}")
    for i in range(2):
        t = IntervalAI()
        t.config.set_level(level)
        ex = t.generate_example()
        out.append(f"## Example {i+1}")
        out.append(f"**Prompt:**")
        out.append("")
        out.append(t.render_prompt(ex.metadata))
        out.append("")
        out.append(f"**Answer:** `{ex.answer}`")
        out.append("")

with open("reasoning_core/tasks/generated/wave0/n11_interval_abstract_interpretation/samples_N11.md", "w") as f:
    f.write("\n".join(out))
print("written")
