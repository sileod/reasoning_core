import random
from pathlib import Path

from reasoning_core.tasks.generated.wave9.ledger_reconciliation.ledger_reconciliation import (
    LedgerReconciliation,
)

SEED = 3238464831


def main():
    random.seed(SEED)
    task = LedgerReconciliation()
    out = Path(__file__).with_name("samples_P069v1.md")
    with open(out, "w") as f:
        for level in (0, 2, 5):
            f.write(f"## Level {level}\n\n")
            cfg = task.config
            cfg.set_level(level)
            task.config = cfg
            for i in range(2):
                x = task.generate_example()
                f.write(f"### Example {i+1}\n\n")
                f.write("**Prompt:**\n\n")
                f.write(task.render_prompt(x.metadata) + "\n\n")
                f.write("**Answer:**\n\n")
                f.write(x.answer + "\n\n")


if __name__ == "__main__":
    main()
