import random
from pathlib import Path

random.seed(1322404106)

from reasoning_core.tasks.generated.wave9.closure_capture_resolution.closure_capture_resolution import (
    ClosureCaptureResolution,
)


def build_level(level):
    task = ClosureCaptureResolution()
    entries = []
    for _ in range(30):
        e = task.generate_example(level=level)
        entries.append(e)
    return entries


def render_example(e):
    prompt = e.metadata["prompt"] if "prompt" in e.metadata else None
    if prompt is None:
        prompt = ClosureCaptureResolution().render_prompt(e.metadata)
    return prompt, e.answer


with open(Path(__file__).with_name("samples_P026v1.md"), "w") as f:
    for level in (0, 2, 5):
        f.write(f"# Level {level}\n\n")
        entries = build_level(level)
        for idx, e in enumerate(entries[:2]):
            prompt, answer = render_example(e)
            f.write(f"## Example {idx + 1}\n\n")
            f.write("**Prompt:**\n\n")
            f.write(prompt + "\n\n")
            f.write("**Answer:** " + answer + "\n\n")
