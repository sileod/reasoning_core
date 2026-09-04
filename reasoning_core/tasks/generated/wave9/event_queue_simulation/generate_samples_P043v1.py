import random
from pathlib import Path
from reasoning_core.tasks.generated.wave9.event_queue_simulation.event_queue_simulation import EventQueueSimulation

SEED = 4052223613


def main():
    random.seed(SEED)
    out = []
    for level in (0, 2, 5):
        task = EventQueueSimulation()
        task.config.set_level(level)
        out.append(f"## Level {level}")
        for i in range(2):
            e = task.generate_example()
            out.append(f"### Example {i+1}")
            out.append(task.render_prompt(e.metadata))
            out.append("Answer:")
            out.append(e.answer)
            out.append("")
    path = Path(__file__).with_name("samples_P043v1.md")
    path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
