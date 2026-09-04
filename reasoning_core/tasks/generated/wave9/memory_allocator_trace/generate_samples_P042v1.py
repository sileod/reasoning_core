import random
from pathlib import Path
from reasoning_core.tasks.generated.wave9.memory_allocator_trace.memory_allocator_trace import MemoryAllocatorTrace

random.seed(103415150)

out = Path(__file__).with_name("samples_P042v1.md")
lines = []
task = MemoryAllocatorTrace()

for level in (0, 2, 5):
    lines.append(f"Level {level}")
    lines.append("")
    cfg = task.config_cls()
    cfg.set_level(level)
    task.config = cfg
    for _ in range(2):
        x = task.generate_example()
        lines.append("Prompt:")
        lines.append(task.render_prompt(x.metadata))
        lines.append("")
        lines.append("Answer:")
        lines.append(x.answer)
        lines.append("")

out.write_text("\n".join(lines))
