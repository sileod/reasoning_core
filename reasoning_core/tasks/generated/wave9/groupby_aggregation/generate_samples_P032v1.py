import random
from pathlib import Path

import reasoning_core.tasks.generated.wave9.groupby_aggregation.groupby_aggregation as mod


def main():
    random.seed(349546629)
    task = mod.GroupbyAggregation()
    out = []
    out.append("# Samples P032v1 - groupby_aggregation")
    out.append("")
    for level in (0, 2, 5):
        out.append("## Level %d" % level)
        out.append("")
        cfg = mod.GroupbyAggregationConfig()
        cfg.apply_difficulty(level)
        task.config = cfg
        for i in range(2):
            e = task.generate_entry()
            out.append("### Example %d (level %d)" % (i + 1, level))
            out.append("")
            out.append("**Prompt:**")
            out.append("")
            out.append(task.render_prompt(e.metadata))
            out.append("")
            out.append("**Answer:** `%s`" % e.answer)
            out.append("")
    path = Path(__file__).with_name("samples_P032v1.md")
    path.write_text("\n".join(out))


if __name__ == "__main__":
    main()
