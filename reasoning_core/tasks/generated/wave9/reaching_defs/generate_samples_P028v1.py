import importlib.util
import random
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("reaching_definitions.py")
_seed_script = None  # noqa


def _load():
    spec = importlib.util.spec_from_file_location(
        "wave9_reaching_definitions_trial", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    random.seed(2905774178)
    mod = _load()
    ReachingDefs = mod.ReachingDefs
    task = ReachingDefs()
    lines = []
    for level in (0, 2, 5):
        lines.append("# Level %d\n" % level)
        task.config.set_level(level)
        for i in range(2):
            e = task.generate_example()
            lines.append("## Example %d\n" % (i + 1))
            lines.append("### Prompt\n")
            lines.append(task.render_prompt(e.metadata) + "\n")
            lines.append("### Answer\n")
            lines.append(e.answer + "\n")
        lines.append("\n")
    Path(__file__).with_name("samples_P028v1.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
