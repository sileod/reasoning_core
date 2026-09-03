import random
from pathlib import Path

from reasoning_core.tasks.generated.wave4.s44_spreadsheet_evaluation.spreadsheet_evaluation import (
    SpreadsheetEvaluation, SpreadsheetEvaluationConfig)


def main():
    random.seed(2063019764)
    out_path = Path(__file__).with_name("samples_S44.md")
    lines = []
    for lvl in [0, 2, 5]:
        lines.append(f"## Level {lvl}")
        lines.append("")
        cfg = SpreadsheetEvaluationConfig()
        cfg.set_level(lvl)
        t = SpreadsheetEvaluation(config=cfg)
        for i in range(2):
            e = t.generate_example()
            lines.append(f"### Example {i + 1}")
            lines.append("")
            prompt = t.render_prompt(e.metadata)
            lines.append("Prompt:")
            lines.append("```")
            lines.append(prompt)
            lines.append("```")
            lines.append("")
            lines.append(f"Answer: {e.answer}")
            lines.append("")
    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
