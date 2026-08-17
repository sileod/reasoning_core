#!/usr/bin/env python3
"""Build the ORPO WARM-UP training set: preference pairs disjoint from every eval leg.

The warm-up is common-mode across arms -- it moves the starting point, not the per-task treatment --
so training on the same SOURCES as the preference eval legs is fine. Training on the same ITEMS is
not. This builder therefore takes tulu3 + ultramix from the pinned revisions and drops any pair whose
prompt already appears in ANY data_cache/*_eval.jsonl, which covers the eval legs, the dev legs
(uf_short, python_dpo) and anything added later without this file needing to know their names.

    python -m reasoning_core.evals.build_orpo_warm --n 6000

Output rows are {prompt, chosen, rejected}; `collection_influence warmup --warm-loss orpo` consumes
them directly. Overlap is reported, not silently tolerated: a build that excluded nothing means the
hashing is wrong, and a build that excluded almost everything means the split is unusable.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, re
from pathlib import Path

DC = Path(os.environ.get("EVAL_DATA_DIR", "data_cache"))
REV = {"allenai/llama-3.1-tulu-3-8b-preference-mixture": "78a6f00785946cd24276c5dd075f83a143a3b1e6",
       "aladinDJ/ultramix-DPO-annotated": "3fb2a270ec7c4756fa518b9452da2ddc967da2a2"}


def _key(text: str) -> str:
    """Whitespace/case-insensitive prompt hash. The eval builders reformat and truncate prompts, so
    an exact-string match would miss items that are the same question rendered differently."""
    return hashlib.sha1(re.sub(r"\s+", " ", (text or "")).strip().lower().encode()).hexdigest()[:16]


def eval_keys() -> set[str]:
    keys = set()
    for f in sorted(DC.glob("*_eval.jsonl")):
        for line in f.read_text(errors="ignore").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            keys.add(_key(r.get("prompt", "")))
            # preference legs carry the prompt inside the shared stem; index the choices too, so a
            # response reused as a distractor cannot come back as a `chosen` here.
            for c in r.get("choices") or ():
                keys.add(_key(c))
    return keys


def tulu3(n, split):
    from datasets import load_dataset
    src = "allenai/llama-3.1-tulu-3-8b-preference-mixture"
    for r in load_dataset(src, split=split, revision=REV[src], streaming=True):
        ch, rj = list(r.get("chosen") or []), list(r.get("rejected") or [])
        if len(ch) >= 2 and len(rj) >= 2 and r.get("prompt"):
            yield str(r["prompt"]), ch[-1]["content"], rj[-1]["content"]


def ultramix(n, split):
    """Widest-margin pairs first is not possible while streaming, so take the pairs whose annotated
    reward gap clears a floor -- same intent, one pass."""
    from datasets import load_dataset
    src = "aladinDJ/ultramix-DPO-annotated"
    for r in load_dataset(src, split=split, revision=REV[src], streaming=True):
        if str(r.get("language") or "EN") != "EN":
            continue
        try:
            gap = float(r["chosen_instruct_reward"]) - float(r["rejected_instruct_reward"])
        except (TypeError, ValueError, KeyError):
            continue
        ch, rj = list(r.get("chosen") or []), list(r.get("rejected") or [])
        if gap >= 1.0 and len(ch) >= 2 and len(rj) >= 2 and r.get("prompt"):
            yield str(r["prompt"]), ch[-1]["content"], rj[-1]["content"]


SOURCES = {"tulu3": tulu3, "ultramix": ultramix}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=["tulu3", "ultramix"], choices=sorted(SOURCES))
    ap.add_argument("--n", type=int, default=6000, help="pairs per source AFTER exclusion")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-chars", type=int, default=2400, help="prompt+response cap, per side")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", default=str(DC / "orpo_warm_train.jsonl"))
    a = ap.parse_args()

    excl = eval_keys()
    print(f"[orpo-warm] {len(excl)} prompt/choice keys held out from {len(list(DC.glob('*_eval.jsonl')))} eval files")
    if not excl:
        raise SystemExit("[orpo-warm] no eval keys found -- refusing to build an unfiltered set")

    rows, stats = [], {}
    for s in a.sources:
        kept = dropped = 0
        for prompt, ch, rj in SOURCES[s](a.n, a.split):
            if kept >= a.n:
                break
            if _key(prompt) in excl or _key(ch) in excl:
                dropped += 1
                continue
            if len(prompt) > a.max_chars or not ch.strip() or not rj.strip() or ch.strip() == rj.strip():
                continue
            rows.append({"prompt": prompt, "chosen": ch[:a.max_chars], "rejected": rj[:a.max_chars],
                         "_source": s})
            kept += 1
        stats[s] = (kept, dropped)
        print(f"[orpo-warm] {s:<9} kept {kept:<6} excluded-as-contaminated {dropped}")

    random.Random(a.seed).shuffle(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"[orpo-warm] {len(rows)} pairs -> {a.out}")


if __name__ == "__main__":
    main()
