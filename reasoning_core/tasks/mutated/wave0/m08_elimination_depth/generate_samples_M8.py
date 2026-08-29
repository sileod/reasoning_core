import random

from reasoning_core.tasks.mutated.wave0.m08_elimination_depth.elimination_depth_chain import \
    EliminationDepthChain


def build_level(level):
    task = EliminationDepthChain()
    task.config.set_level(level)
    return task


def generate_samples():
    random.seed(3396779050)
    lines = []
    for level in (0, 2, 5):
        task = build_level(level)
        lines.append(f"# Level {level}\n")
        for i in range(2):
            x = task.generate_example()
            prompt = task.render_prompt(x.metadata)
            lines.append(f"## Example {i + 1}\n")
            lines.append(f"**Prompt:**\n```\n{prompt}\n```\n")
            lines.append(f"**Answer:**\n```\n{x.answer}\n```\n")
            c = x.metadata
            lines.append(
                f"_diagnostic: target={c['target_depth']} solver={c['diagnostic_depth']} "
                f"num_vars={c['num_vars']}_\n\n"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    content = generate_samples()
    out = "reasoning_core/tasks/mutated/wave0/m08_elimination_depth/samples_M8.md"
    with open(out, "w") as f:
        f.write(content)
    print(content)
