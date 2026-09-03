import random
from pathlib import Path

from reasoning_core.tasks.generated.wave5.s56_critical_path.critical_path_task import CriticalPath


def main():
    random.seed(2778460392)
    t = CriticalPath()
    out = []
    for level in [0, 2, 5]:
        cfg = CriticalPath().config_cls()
        cfg.set_level(level)
        t.config = cfg
        out.append(f"## Level {level}")
        for _ in range(2):
            e = t.generate_entry()
            out.append("### Prompt")
            out.append(t.render_prompt(e.metadata))
            out.append("### Answer")
            out.append(e.answer)
            out.append("")
    path = Path(__file__).with_name("samples_S56.md")
    path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
