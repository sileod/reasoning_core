import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.firewall_rule_shadowing.firewall_rule_shadowing import (
    FirewallRuleShadowing,
)

random.seed(2546520183)
task = FirewallRuleShadowing()
out = []
for level in (0, 2, 5):
    task.config.set_level(level)
    out.append("## Level %d" % level)
    for k in range(2):
        ex = task.generate_example()
        out.append("### Example %d" % (k + 1))
        out.append("**Prompt:**")
        out.append(task.render_prompt(ex.metadata))
        out.append("")
        out.append("**Answer:**")
        out.append(ex.answer)
        out.append("")
Path(__file__).with_name("samples_P052v1.md").write_text("\n".join(out))
print("wrote samples")
