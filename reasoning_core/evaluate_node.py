"""
evaluate_node.py
────────────────
Two modes:

  generate  — create a fixed evaluation set and save to JSON
  evaluate  — load the fixed set, run a model, report per-tier accuracy

Usage
─────
  # Step 1: generate once and freeze the dataset
  python evaluate_node.py generate --db functions.db --n 500 --out eval_set.json

  # Step 2: evaluate any model
  python evaluate_node.py evaluate --dataset eval_set.json --model random
  python evaluate_node.py evaluate --dataset eval_set.json --model majority

  # To plug in a real model, implement ModelInterface and pass --model custom

Dataset JSON schema
───────────────────
  {
    "meta": {
      "generated_at": "...",
      "n_examples": 500,
      "db_path": "...",
      "config": {...}
    },
    "examples": [
      {
        "id": 0,
        "difficulty": "execution" | "semantic" | "type-only",
        "answer": "function_name",
        "candidates": [{"name": "...", "is_correct": true/false}, ...],
        "prompt_eval":     "...",   # no exec hint (eval mode)
        "prompt_train":    "...",   # with exec hint (training mode)
        "cot":             "...",
        "code":            "...",
        "masked_type":     "...",
        "masked_module":   "..."
      },
      ...
    ]
  }

Accuracy report
───────────────
  Per-tier accuracy + overall.
  The "gap" (execution accuracy - type-only accuracy) is your core empirical
  claim: it shows models fail on semantically hard examples more than easy ones.
"""

import json
import random
import argparse
from datetime import datetime
from collections import defaultdict, Counter

# Import from the task file
from program_composition_task import (
    NodeCompletion,
    ProgramCompositionCfg,
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dataset(
    db_path    : str,
    n          : int,
    out_path   : str,
    depth      : int = 3,
    seed       : int = 0,
    use_sandbox: bool = True,
):
    cfg = ProgramCompositionCfg(
        db_path     = db_path,
        max_depth   = depth,
        seed        = seed,
        use_sandbox = use_sandbox,
    )
    task     = NodeCompletion(config=cfg)
    examples = []
    skipped  = 0
    seen_hashes = set()

    print(f"Generating {n} examples (sandbox={'on' if use_sandbox else 'off'}) ...")

    attempts = 0
    while len(examples) < n:
        attempts += 1
        if attempts > n * 20:
            print(f"Warning: gave up after {attempts} attempts, only {len(examples)} generated.")
            break
        try:
            ex = task.generate_example()
        except RuntimeError:
            skipped += 1
            continue

        h = ex.metadata.get("code_hash", "")
        if h in seen_hashes:
            skipped += 1
            continue
        seen_hashes.add(h)

        examples.append({
            "id":            len(examples),
            "difficulty":    ex.metadata["difficulty"],
            "answer":        ex.answer,
            "candidates":    ex.metadata["candidates"],
            "prompt_eval":   task.prompt(ex.metadata, include_exec_hint=False),
            "prompt_train":  task.prompt(ex.metadata, include_exec_hint=True),
            "cot":           ex.cot,
            "code":          ex.metadata["code"],
            "masked_type":   ex.metadata["masked_type"],
            "masked_module": ex.metadata["masked_module"],
            "context_desc":  ex.metadata.get("context_desc", ""),
            "exec_desc":     ex.metadata.get("exec_desc", ""),
        })

        if len(examples) % 50 == 0:
            dist = Counter(e["difficulty"] for e in examples)
            print(f"  {len(examples)}/{n}  difficulty: {dict(dist)}")

    dist = Counter(e["difficulty"] for e in examples)
    dataset = {
        "meta": {
            "generated_at"  : datetime.now().isoformat(),
            "n_examples"    : len(examples),
            "n_attempted"   : attempts,
            "n_skipped"     : skipped,
            "db_path"       : db_path,
            "difficulty_dist": dict(dist),
            "config": {
                "depth"      : depth,
                "seed"       : seed,
                "use_sandbox": use_sandbox,
            },
        },
        "examples": examples,
    }

    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nSaved {len(examples)} examples to {out_path}")
    print(f"Difficulty distribution: {dict(dist)}")
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class ModelInterface:
    """Override predict() to plug in a real model."""
    name = "base"

    def predict(self, prompt: str, candidates: list) -> str:
        """Return a candidate name or letter (A, B, C, ...)."""
        raise NotImplementedError


class RandomModel(ModelInterface):
    """Baseline: random choice among candidates."""
    name = "random"

    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def predict(self, prompt: str, candidates: list) -> str:
        return self.rng.choice(candidates)["name"]


class MajorityModel(ModelInterface):
    """
    Baseline: always pick the candidate whose name appears most often
    across all examples (frequency bias exploit).
    """
    name = "majority"

    def __init__(self, examples: list):
        counts: Counter = Counter()
        for ex in examples:
            for c in ex["candidates"]:
                counts[c["name"]] += 1
        self._most_common = counts.most_common(1)[0][0]

    def predict(self, prompt: str, candidates: list) -> str:
        names = [c["name"] for c in candidates]
        return self._most_common if self._most_common in names else names[0]


class FirstCandidateModel(ModelInterface):
    """Baseline: always pick candidate A (position bias)."""
    name = "first_candidate"

    def predict(self, prompt: str, candidates: list) -> str:
        return candidates[0]["name"]


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def score_prediction(prediction: str, example: dict) -> float:
    """
    Returns 1.0 if correct, 0.0 otherwise.
    Accepts function name or letter (A, B, C, ...).
    """
    ans        = str(prediction).strip()
    candidates = example["candidates"]
    correct    = example["answer"]

    if len(ans) == 1 and ans.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        idx = ord(ans.upper()) - ord("A")
        if 0 <= idx < len(candidates):
            return float(candidates[idx]["name"] == correct)

    return float(ans.strip().lower().strip("`") == correct.strip().lower())


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(
    dataset_path: str,
    model        : ModelInterface,
    prompt_mode  : str = "eval",   # "eval" (no hint) or "train" (with hint)
    verbose      : bool = False,
):
    with open(dataset_path) as f:
        dataset = json.load(f)

    examples = dataset["examples"]
    prompt_key = "prompt_eval" if prompt_mode == "eval" else "prompt_train"

    # Per-tier accumulators
    tier_scores: dict[str, list] = defaultdict(list)
    all_scores  : list[float] = []

    for ex in examples:
        prompt     = ex[prompt_key]
        candidates = ex["candidates"]
        pred       = model.predict(prompt, candidates)
        score      = score_prediction(pred, ex)
        tier       = ex["difficulty"]
        tier_scores[tier].append(score)
        all_scores.append(score)

        if verbose:
            status = "✓" if score == 1.0 else "✗"
            print(f"  [{tier:10s}] {status}  pred={pred!r}  correct={ex['answer']!r}")

    # ── Report ─────────────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  Model: {model.name}   |   prompt_mode: {prompt_mode}")
    print(f"  Dataset: {dataset_path}  ({len(examples)} examples)")
    print(f"{'─'*55}")

    tier_order = ["execution", "semantic", "type-only"]
    for tier in tier_order:
        scores = tier_scores.get(tier, [])
        if not scores:
            continue
        acc = sum(scores) / len(scores)
        print(f"  {tier:12s}  n={len(scores):4d}   acc={acc:.3f}  ({acc*100:.1f}%)")

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"{'─'*55}")
    print(f"  {'OVERALL':12s}  n={len(all_scores):4d}   acc={overall:.3f}  ({overall*100:.1f}%)")

    # The gap is the core empirical claim
    exec_acc   = (sum(tier_scores["execution"]) / len(tier_scores["execution"])
                  if tier_scores["execution"] else None)
    type_acc   = (sum(tier_scores["type-only"]) / len(tier_scores["type-only"])
                  if tier_scores["type-only"] else None)
    if exec_acc is not None and type_acc is not None:
        gap = type_acc - exec_acc
        print(f"\n  Difficulty gap (type-only acc − execution acc): {gap:+.3f}")
        print(f"  (positive gap = execution examples are harder, as expected)")
    print(f"{'═'*55}\n")

    return {
        "model": model.name,
        "prompt_mode": prompt_mode,
        "overall": overall,
        "by_tier": {tier: sum(s) / len(s) for tier, s in tier_scores.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NodeCompletion evaluation")
    sub = parser.add_subparsers(dest="mode", required=True)

    gen_p = sub.add_parser("generate", help="Generate fixed evaluation dataset")
    gen_p.add_argument("--db",         default="functions.db")
    gen_p.add_argument("--n",          type=int, default=500)
    gen_p.add_argument("--out",        default="eval_set.json")
    gen_p.add_argument("--depth",      type=int, default=3)
    gen_p.add_argument("--seed",       type=int, default=0)
    gen_p.add_argument("--no-sandbox", action="store_true")

    ev_p = sub.add_parser("evaluate", help="Evaluate a model on a fixed dataset")
    ev_p.add_argument("--dataset",      default="eval_set.json")
    ev_p.add_argument("--model",
                      choices=["random", "majority", "first"],
                      default="random")
    ev_p.add_argument("--prompt-mode",
                      choices=["eval", "train"],
                      default="eval",
                      help="eval=no exec hint, train=with exec hint")
    ev_p.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.mode == "generate":
        generate_dataset(
            db_path    = args.db,
            n          = args.n,
            out_path   = args.out,
            depth      = args.depth,
            seed       = args.seed,
            use_sandbox= not args.no_sandbox,
        )

    elif args.mode == "evaluate":
        with open(args.dataset) as f:
            dataset = json.load(f)
        examples = dataset["examples"]

        if args.model == "random":
            model = RandomModel()
        elif args.model == "majority":
            model = MajorityModel(examples)
        elif args.model == "first":
            model = FirstCandidateModel()

        evaluate(
            dataset_path = args.dataset,
            model        = model,
            prompt_mode  = args.prompt_mode,
            verbose      = args.verbose,
        )


if __name__ == "__main__":
    main()