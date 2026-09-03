import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.schedule_recoverability import \
    schedule_recoverability as mod


def main():
    random.seed(1669468372)
    task = mod.ScheduleRecoverability()
    out = []
    for level in (0, 2, 5):
        out.append(f"## Level {level}\n")
        for _ in range(2):
            ex = task.generate_example(level=level)
            out.append("Prompt:")
            out.append(ex.prompt)
            out.append("")
            out.append("Answer:")
            out.append(ex.answer)
            out.append("")
        out.append("")
    path = Path(__file__).with_name("samples_P032v2.md")
    path.write_text("\n".join(out))


if __name__ == "__main__":
    main()
