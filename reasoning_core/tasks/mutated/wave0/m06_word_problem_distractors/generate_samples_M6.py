import random
import sys
sys.path.insert(0, ".")

from reasoning_core.tasks.mutated.wave0.m06_word_problem_distractors.word_problem_distractors import WordProblemDistractors

seed = 2364728918
random.seed(seed)

task = WordProblemDistractors()
lines = [
    f"# WordProblemDistractors samples (seed {seed})",
    "",
    "Two complete prompt/answer examples per level 0, 2, 5.",
    "",
]
for level in (0, 2, 5):
    task.config.set_level(level)
    for i in range(2):
        problem = task.generate_entry()
        prompt = task.render_prompt(problem.metadata)
        lines.append("---")
        lines.append(f"### level {level} example {i+1}")
        lines.append("")
        lines.append("**Prompt:**")
        lines.append("")
        lines.append(prompt)
        lines.append("")
        lines.append("**Answer:**")
        lines.append("")
        lines.append(problem.answer)
        lines.append("")
        lines.append(f"proof_core_size={problem.metadata.proof_core_size} "
                     f"distractors={problem.metadata.distractor_count} "
                     f"unit={problem.metadata.unit}")
        lines.append("")

with open("reasoning_core/tasks/mutated/wave0/m06_word_problem_distractors/samples_M6.md", "w") as fh:
    fh.write("\n".join(lines))
print("wrote", len(lines), "lines")
