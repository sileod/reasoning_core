import random

from reasoning_core.tasks.generated.wave0.n07_max_flow_min_cut.max_flow_min_cut import MaxFlowMinCut

random.seed(3487141474)

lines = []
for level in (0, 2, 5):
    lines.append(f"# Level {level}")
    t = MaxFlowMinCut()
    t.config.set_level(level)
    for _ in range(2):
        entry = t.generate_example()
        prompt = t.render_prompt(entry.metadata)
        lines.append(f"## Prompt\n{prompt}")
        lines.append(f"## Answer\n{entry.answer}")
        lines.append("")

out_path = __file__.rsplit("/", 1)[0] + "/samples_N7.md"
with open(out_path, "w") as f:
    f.write("\n".join(lines))
print("wrote samples_N7.md")
