#!/usr/bin/env python3
"""Build MCQ contrastive eval legs from public NLU datasets (BoolQ, SIQA, WiC, PIQA, ReClor).

These are *contrast* legs, not preference legs: each item is one prompt plus a small fixed choice set
with exactly one gold, scored by mean token logprob per candidate. That makes them directly comparable
to arc_easy / balanced_copa, and it is the shape that survived the leak critique -- the choices are
NOT listed in the prompt, so there is no option-menu to format-match against.

    python -m reasoning_core.evals.build_contrastive_evals --tasks boolq siqa wic piqa reclor

Revisions are PINNED. `load_dataset(..., revision=None)` silently resolves to whatever snapshot the
hub serves that day, which has already produced one stale-pool incident in this project.

Every build prints the three gates before the file is usable as a leg:
  1 GAMEABLE   -- answer leaked into the prompt; a shortest-answer picker beating chance
  2 INFORMATIVE-- a base model must score above chance (measured separately, see probe_leg_*)
  3 SENSITIVE  -- must move across aux arms (only measurable after a run)
Gate 1 is checked here because it is a property of the DATA and is cheap; a leg that fails it should
never reach a battery.
"""
from __future__ import annotations
import argparse, json, os, random, re
from pathlib import Path

DC = Path(os.environ.get("EVAL_DATA_DIR", "data_cache"))

REVISIONS = {
    "google/boolq": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
    "lighteval/siqa": "54c6a1f8cb6daf4f5abf24a601852612fb35eb25",
    "aps/super_glue": "3de24cf8022e94f4ee4b9d55a6f539891524d646",
    # ybisk/piqa is a script dataset (needs trust_remote_code and will not pin); baber/piqa is
    # the same data as parquet, so it loads under a pinned revision like everything else.
    "baber/piqa": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
    "tasksource/reclor": "0414b39bfc39b2d555b48fa245fc198744c46215",
}


def _load(name, *args, **kw):
    from datasets import load_dataset
    rev = REVISIONS.get(name)
    try:
        return load_dataset(name, *args, revision=rev, **kw)
    except Exception as e:
        # A wrong pin must NOT quietly become "whatever the hub serves today" -- that is the exact
        # staleness this pinning exists to prevent. Fail, and print the CURRENT sha to paste back in.
        try:
            from huggingface_hub import HfApi
            cur = HfApi().dataset_info(name).sha
        except Exception:
            cur = "<could not resolve>"
        raise SystemExit(f"[build] revision {rev} unusable for {name} ({type(e).__name__}). "
                         f"Current hub sha is {cur} -- update REVISIONS, do not run unpinned.")


def _row(prompt, gold, distractors, rng):
    choices = [gold] + [d for d in distractors if d and d != gold]
    if len(choices) < 2:
        return None
    rng.shuffle(choices)
    return {"prompt": prompt.strip() + "\nAnswer:", "answer": gold,
            "choices": choices, "answer_idx": choices.index(gold)}


def build_boolq(n, rng):
    """Passage + question -> yes/no. Contrast is minimal by construction: same passage, two answers."""
    for r in _load("google/boolq", split="validation").select(range(min(n, 3270))):
        yield _row(f"{r['passage']}\nQuestion: {r['question']}?", "yes" if r["answer"] else "no",
                   ["no" if r["answer"] else "yes"], rng)


def build_siqa(n, rng):
    """Social commonsense: context+question, 3 plausible answers. The distractors are human-written."""
    d = _load("lighteval/siqa", split="validation")
    for r in d.select(range(min(n, len(d)))):
        opts = [r["answerA"], r["answerB"], r["answerC"]]
        gold = opts[int(r["label"]) - 1]
        yield _row(f"{r['context']}\nQuestion: {r['question']}", gold,
                   [o for o in opts if o != gold], rng)


def build_wic(n, rng):
    """Word-in-context: same word, two sentences, same sense or not. Minimal pair by design."""
    d = _load("aps/super_glue", "wic", split="validation")
    for r in d.select(range(min(n, len(d)))):
        p = (f"Sentence 1: {r['sentence1']}\nSentence 2: {r['sentence2']}\n"
             f"Question: is '{r['word']}' used with the same meaning in both sentences?")
        gold = "yes" if r["label"] == 1 else "no"
        yield _row(p, gold, ["no" if gold == "yes" else "yes"], rng)


def build_piqa(n, rng):
    """Physical commonsense: a goal and two procedures, one workable. Length-matched by construction."""
    d = _load("baber/piqa", split="validation")
    for r in d.select(range(min(n, len(d)))):
        opts = [r["sol1"], r["sol2"]]
        gold = opts[int(r["label"])]          # label is a STRING in this mirror
        yield _row(f"Goal: {r['goal']}", gold, [o for o in opts if o != gold], rng)


def build_reclor(n, rng):
    """Logical reading comprehension (LSAT/GMAT). The hardest of the five; 4 options."""
    d = _load("tasksource/reclor", split="validation")
    for r in d.select(range(min(n, len(d)))):
        opts = list(r["answers"])
        gold = opts[int(r["label"])]
        yield _row(f"{r['context']}\nQuestion: {r['question']}", gold,
                   [o for o in opts if o != gold], rng)


BUILDERS = {"boolq": build_boolq, "siqa": build_siqa, "wic": build_wic,
            "piqa": build_piqa, "reclor": build_reclor}


def gate1(rows):
    """GAMEABLE checks on the data alone: answer leaked into the prompt, and shortest-picker excess."""
    # For a binary yes/no leg the substring test is vacuous -- "no" occurs inside ordinary words in
    # any passage -- so it is skipped there and the shortest-picker excess carries gate 1 instead.
    binary = all(set(x.lower() for x in r["choices"]) == {"yes", "no"} for r in rows)
    leak = 0.0 if binary else sum(r["answer"].lower() in r["prompt"].lower() for r in rows) / len(rows)
    short = sum(min(r["choices"], key=len) == r["answer"] for r in rows) / len(rows)
    chance = sum(1 / len(r["choices"]) for r in rows) / len(rows)
    dlen = (sum(len(r["answer"]) for r in rows) / len(rows)
            - sum(sum(len(c) for c in r["choices"]) / len(r["choices"]) for r in rows) / len(rows))
    return {"leak": leak, "binary": binary, "shortest_picker": short, "chance": chance,
            "shortest_excess_pt": 100 * (short - chance), "gold_len_delta_chars": dlen}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=sorted(BUILDERS))
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--max-prompt-chars", type=int, default=1400,
                    help="drop items whose prompt is too long for the battery's max_length")
    ap.add_argument("--seed", type=int, default=43)
    a = ap.parse_args()
    DC.mkdir(parents=True, exist_ok=True)
    for t in a.tasks:
        rng = random.Random(a.seed)
        rows = [r for r in BUILDERS[t](a.n * 2, rng) if r]
        rows = [r for r in rows if len(r["prompt"]) <= a.max_prompt_chars][:a.n]
        if not rows:
            print(f"[build] {t}: NO rows survived; skipped"); continue
        out = DC / f"{t}_eval.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        g = gate1(rows)
        flag = ("  <-- FAILS GATE 1" if g["leak"] > 0.10 or g["shortest_excess_pt"] > 10 else "")
        print(f"[build] {t:<7} n={len(rows):<5} leak={'n/a (binary)' if g['binary'] else format(g['leak'],'.1%')} "
              f"shortest-picker={g['shortest_picker']:.1%} (chance {g['chance']:.1%}, "
              f"excess {g['shortest_excess_pt']:+.1f}pt) gold_len_delta={g['gold_len_delta_chars']:+.0f}ch"
              f" -> {out}{flag}")


if __name__ == "__main__":
    main()
