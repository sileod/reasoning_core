#!/usr/bin/env python
import argparse
import ast
import functools
import hashlib
import inspect
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from tqdm.auto import tqdm

from reasoning_core import get_task, list_tasks
from reasoning_core.registry import _task_to_module_map
from reasoning_core.template import _strip_docstrings


ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "reasoning_core" / "tasks"
CATEGORY_ORDER = [
    "arithmetics", "equation_system", "combinatorics", "function_manipulation",
    "math_lean", "math_tptp", "math_geometry", "math_metamath", "binding",
    "probabilistic_reasoning", "causal_reasoning", "logic_semantics",
    "logic_depth", "logic_derivation", "planning", "set_operations", "sequential_induction",
    "qstr", "grid_navigation", "tracking", "belief_tracking", "coreference",
    "constraint_satisfaction", "graph_operations", "regex",
    "formal_analogies", "grammar", "knowledge", "table_qa",
    "string_transduction", "game_playing", "qualitative_causal_reasoning",
    "code_analysis", "code_execution", "code_program_synthesis",
]

SINGLE_EXAMPLE_TASKS = {
    "tptp_entailment",
    "theorem_premise_selection",
    "proof_reconstruction",
    "parsing",
    "tptp_consistency_repair",
}

GITHUB_BASE = "https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks"
README_MENU_RE = re.compile(
    r"(\[GALLERY\]\([^)]+\)[^\n]*\n\n)([^\n]*)(\n\n+## Task authoring guidelines)",
)


def fence(text):
    text = str(text)
    max_run = max((len(m) for m in re.findall(r"`+", text)), default=2)
    ticks = "`" * max(max_run + 1, 3)
    return f"{ticks}\n{text}\n{ticks}"


def slug(name):
    return re.sub(r"[^\w\s-]", "", name.lower()).replace(" ", "-")


def task_module(name):
    return _task_to_module_map[name][0]


def task_source(name):
    # Tasks under reasoning_core/tasks/generated/ report their module as "generated.<name>", so the
    # dotted path has to become a directory separator. Joining it verbatim yields
    # tasks/generated.<name>.py, which does not exist, and the build dies on the first such task.
    return TASKS_DIR.joinpath(*task_module(name).split(".")).with_suffix(".py")


@functools.lru_cache(maxsize=None)
def _source_behavior_hash(path_str):
    # Canonicalisation is imported, not repeated: a second copy silently drifts from the one that
    # stamps the rows, and the hash is only meaningful if every producer computes it identically.
    from reasoning_core.template import _stable_dump
    tree = ast.parse(Path(path_str).read_text(encoding="utf-8"), filename=path_str)
    return hashlib.sha1(_stable_dump(_strip_docstrings(tree)).encode()).hexdigest()[:16]


def source_behavior_hash(path):
    # Many tasks share one module file; memoize per file so the changed-task
    # scan parses each module once, not once per task.
    return _source_behavior_hash(str(Path(path)))


def source_modified_date(path):
    return datetime.fromtimestamp(Path(path).stat().st_mtime).strftime("%Y.%m.%d")


def category_rank_name(name):
    try:
        return CATEGORY_ORDER.index(task_module(name))
    except ValueError:
        return len(CATEGORY_ORDER)


def sort_task_names(names):
    registry_rank = {name: i for i, name in enumerate(list_tasks())}
    return sorted(
        names,
        key=lambda name: (category_rank_name(name), registry_rank.get(name, 10**9)),
    )


def pick_example(task, batch_size, cache=False, refresh_cache=False):
    if batch_size <= 1:
        return task.generate_example()
    if cache:
        batch = task.validate(n_samples=batch_size, cache=True,
                              refresh=refresh_cache)
    else:
        batch = task.generate_balanced_batch(batch_size=batch_size)
    return sorted(batch, key=lambda x: len(x.prompt) - len(str(x.answer)))[0]


def load_taskrow_examples(task_names, cache_path):
    if not cache_path:
        return OrderedDict()
    from task_diagnostics.cache import load_task_rows

    wanted = set(task_names)
    candidates = {}
    for row in load_task_rows(path=cache_path, tasks=wanted):
        if row.task not in wanted:
            continue
        example = SimpleNamespace(prompt=row.prompt, answer=row.answer)
        rank = (row.level, len(row.prompt) - len(str(row.answer)), len(row.prompt))
        if row.task not in candidates or rank < candidates[row.task][0]:
            candidates[row.task] = (rank, example)
    return OrderedDict(
        (name, candidates[name][1])
        for name in task_names
        if name in candidates
    )


def source_link(task):
    rel_path = inspect.getfile(task.__class__).split("/tasks")[-1]
    return f"{GITHUB_BASE}{rel_path}"


def task_summary(task):
    return str(getattr(task, "summary", "") or "").strip()


def task_metadata_block(name, task=None, include_summaries=True):
    if task is None:
        task = get_task(name)
    if not include_summaries:
        return ""
    summary = task_summary(task)
    return f"{summary}\n\n" if summary else ""


def behavior_marker(name):
    return f"<!-- behavior-hash: {source_behavior_hash(task_source(name))} -->"


def build_examples(tasks, cache=False, refresh_cache=False, allow_missing=False):
    examples = OrderedDict()
    failures = []
    for task in tqdm(tasks):
        name = task.task_name
        if name in examples:
            continue
        print(name)
        try:
            batch_size = 1 if name in SINGLE_EXAMPLE_TASKS else 4
            examples[name] = pick_example(task, batch_size, cache,
                                          refresh_cache)
        except Exception as e:
            failures.append((name, e))
            print(f"{name}: {e}")
    if failures and not allow_missing:
        names = ", ".join(name for name, _ in failures)
        raise RuntimeError(f"failed to build gallery examples for: {names}")
    return examples


def section_text(name, example, include_summaries=True):
    task = get_task(name)
    return (
        f"## [{name}]({source_link(task)})\n\n"
        f"{behavior_marker(name)}\n\n"
        f"{task_metadata_block(name, task, include_summaries)}"
        f"**Prompt:**\n{fence(example.prompt)}\n\n"
        f"**Answer:**\n{fence(example.answer)}"
    )


def normalize_section(name, section, include_summaries=True):
    section = re.sub(r"\n- hash: `[^`]*`\n", "\n", section)
    section = re.sub(r"\n- modified: [^\n]*\n", "\n", section)
    section = re.sub(r"\n- last modified: [^\n]*\n", "\n", section)
    section = re.sub(r"\n<!-- behavior-hash: [^>]* -->\n", "\n", section)
    header_match = re.match(r"^(## [^\n]+)", section)
    prompt_start = section.find("**Prompt:**")
    if not header_match or prompt_start == -1:
        return section.strip()
    return (
        f"{header_match.group(1)}\n\n"
        f"{behavior_marker(name)}\n\n"
        f"{task_metadata_block(name, include_summaries=include_summaries)}"
        f"{section[prompt_start:].strip()}"
    )


def changed_tasks(task_names, existing_sections, refresh=False):
    if refresh:
        return task_names
    changed = []
    for name in task_names:
        section = existing_sections.get(name, "")
        current_marker = behavior_marker(name)
        date_marker = f"- last modified: {source_modified_date(task_source(name))}"
        legacy_hash_marker = f"- hash: `{source_behavior_hash(task_source(name))}`"
        if not section:
            changed.append(name)
            continue
        if current_marker in section:
            continue
        if "<!-- behavior-hash:" in section:
            changed.append(name)
            continue
        if "- hash:" in section:
            if legacy_hash_marker not in section:
                changed.append(name)
            continue
        if "- last modified:" in section or "- modified:" in section:
            if date_marker not in section:
                changed.append(name)
            continue
    return changed


def read_sections(path):
    path = Path(path)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"^## \[([^\]]+)\].*$", text, re.M))
    sections = {}
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section.endswith("---"):
            section = section[:-3].rstrip()
        sections[match.group(1)] = section
    return sections


def write_gallery(task_names, examples, out_path, existing_sections=None,
                  include_summaries=True):
    out_path = Path(out_path)
    existing_sections = existing_sections or {}
    with out_path.open("w", encoding="utf-8") as f:
        menu = " · ".join(f"[`{t}`](#{slug(t)})" for t in task_names)
        f.write(f"# 📖 Task Gallery\n\n{len(task_names)} tasks\n\n{menu}\n\n---\n\n")
        for index, name in enumerate(task_names):
            if name in examples:
                section = section_text(name, examples[name], include_summaries)
            else:
                section = normalize_section(
                    name, existing_sections[name], include_summaries
                )
            suffix = "\n\n---\n" if index == len(task_names) - 1 else "\n\n---\n\n"
            f.write(f"{section}{suffix}")


def gallery_menu(task_names):
    return " · ".join(f"[`{name}`](GALLERY.md#{slug(name)})" for name in task_names)


def refresh_readme_menu(task_names, readme_path):
    readme_path = Path(readme_path)
    text = readme_path.read_text(encoding="utf-8")
    menu = gallery_menu(task_names)
    new_text, n = README_MENU_RE.subn(rf"\1{menu}\3", text, count=1)
    if n != 1:
        raise RuntimeError(
            f"could not find README gallery menu block in {readme_path}"
        )
    new_text = re.sub(
        r"Browse all \[\d+ task examples\]\(GALLERY\.md\)\.",
        f"Browse all [{len(task_names)} task examples](GALLERY.md).",
        new_text,
        count=1,
    )
    readme_path.write_text(new_text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="GALLERY.md")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--update-readme", choices=["auto", "always", "never"],
                        default="auto",
                        help=("Refresh README gallery menu. auto updates only for "
                              "full GALLERY.md builds."))
    parser.add_argument("--no-cache", action="store_true",
                        help="Use generate_balanced_batch instead of cached validation examples.")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-summaries", action="store_true",
                        help="Omit task summary text from gallery sections.")
    parser.add_argument("--taskrow-cache", default=None,
                        help=("Read examples from a diagnostics TaskRow cache "
                              "(task_diagnostics/cache/task_rows/<cache_id>) "
                              "before generating missing examples."))
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--tasks", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    names = args.tasks or list_tasks()
    task_names = sort_task_names(names)
    existing_sections = read_sections(args.out)
    tasks_to_build = changed_tasks(task_names, existing_sections, args.refresh_cache)
    examples = load_taskrow_examples(tasks_to_build, args.taskrow_cache)
    missing_from_taskrow_cache = [
        name for name in tasks_to_build if name not in examples
    ]
    generated = build_examples(
        (get_task(name) for name in missing_from_taskrow_cache),
        cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
        allow_missing=args.allow_missing,
    )
    examples.update(generated)
    missing = [name for name in task_names if name not in examples and name not in existing_sections]
    if missing:
        if args.allow_missing:
            task_names = [name for name in task_names if name not in missing]
        else:
            raise RuntimeError(f"no generated or existing gallery section for: {', '.join(missing)}")
    write_gallery(
        task_names,
        examples,
        args.out,
        existing_sections,
        include_summaries=not args.no_summaries,
    )
    update_readme = args.update_readme == "always" or (
        args.update_readme == "auto" and args.tasks is None and
        Path(args.out).name == "GALLERY.md"
    )
    if update_readme:
        refresh_readme_menu(task_names, args.readme)


if __name__ == "__main__":
    main()
