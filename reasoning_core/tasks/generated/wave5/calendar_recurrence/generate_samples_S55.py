import importlib.util
import random
from pathlib import Path

SEED = 342837742

MOD_FILE = Path(__file__).with_name("s55_calendar_recurrence.py")
SPEC = importlib.util.spec_from_file_location("s55_calendar_recurrence", MOD_FILE)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def main():
    random.seed(SEED)
    task = MOD.CalendarRecurrence()
    lines = []
    for level in (0, 2, 5):
        cfg = task.config_cls()
        cfg.set_level(level)
        task.config = cfg
        lines.append("## Level {}".format(level))
        for _ in range(2):
            ex = task.generate_example()
            lines.append(task.render_prompt(ex.metadata))
            lines.append("Answer: " + ex.answer)
            lines.append("")
    out = Path(__file__).with_name("samples_S55.md")
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
