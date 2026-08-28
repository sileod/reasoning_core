import random

random.seed(2431745573)

from reasoning_core.tasks.generated.wave0.n08_bipartite_matching.bipartite_matching import BipartiteMatching

out_path = "reasoning_core/tasks/generated/wave0/n08_bipartite_matching/samples_N8.md"
lines = ["# samples_N8", ""]
for L in (0, 2, 5):
    t = BipartiteMatching()
    t.config.set_level(L)
    for i in range(2):
        ex = t.generate_example()
        lines.append(f"## Level {L} example {i+1}")
        lines.append("")
        lines.append("### Prompt")
        lines.append("")
        lines.append(t.render_prompt(ex.metadata))
        lines.append("")
        lines.append("### Answer")
        lines.append("")
        lines.append(ex.answer)
        lines.append("")

with open(out_path, "w") as f:
    f.write("\n".join(lines))
print("wrote", out_path)
