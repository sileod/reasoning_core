#!/usr/bin/env python3
"""collection_influence.py — one influence number per task COLLECTION (rc / rgym / pw / synlogic).

WHY a collection-level score (not the mean of per-task deltas):
    Collections differ in task count and composition (rc=50, rgym~94, synlogic=18, pw=10), so a
    mean-of-per-task-deltas is confounded — it answers "average task", not "this collection". Training
    on the WHOLE collection pooled at a fixed aux budget (MIX_AUX) is exactly how the data is deployed,
    and yields one apples-to-apples ΔNLL per eval leg that IS comparable across collections. This is the
    figure for the paper's collection table.

HOW (reuse, no new training code): a thin driver over per_task_influence.py's built-in COLLECTION mode
    (`COLLECTION=<name>` env → one pooled aux arm `__COLLECTION__` vs the main-only baseline, same paired
    seed, same legs). Each collection is just `AUX_DATASET=rc` + its TaskRow cache. Output lands as
    per_task_results/influence<...>_COLL-<name>_S<seed>_T<steps>_M<mix>_<main>_<init>.json.

USAGE
    # run (serial; point each collection at its cache):
    python -m task_diagnostics.collection_influence run \\
        --model allenai/OLMo-1B-0724-hf --run-tag OLMO1B \\
        --collections rc=FULLROSTER_1783960035,rgym=41944106a49b,synlogic=1b7baa44062a,pw=b8058f4c1f3f \\
        --env LR=2e-5 BATCH=1 GRAD_ACCUM=8 GRAD_CKPT=1 OPTIM=adamw_torch
    # summarize (scan results, emit the table):
    python -m task_diagnostics.collection_influence summarize --run-tag OLMO1B
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("OUT_DIR") or REPO / "per_task_results")
ENGINE = str(REPO / "task_diagnostics" / "per_task_influence.py")
CACHE_ROOT = REPO / "task_diagnostics" / "cache" / "task_rows"

# The 6-leg global score (mean % NLL reduction). mmlu legs use the cloze form (matches the frozen roster).
LEGS = ["bbh", "mmlu_math_cloze", "mmlu_logic_cloze", "mbpp", "fw", "dolci"]

# Per-subject MMLU (build_mmlu_subjects.py) logged on every run in BOTH forms — cloze (answer-text NLL,
# format-fair) + letter (options-in-prompt, gold-letter NLL) — so math/logic transfer reads per subject
# and a broader math macro is reconstructed offline. Engine skips gracefully if a jsonl is absent.
MMLU_SUBJECTS = ["abstract_algebra", "college_mathematics", "elementary_mathematics",
                 "high_school_mathematics", "high_school_statistics", "formal_logic"]


def _resolve_cache(spec: str) -> str:
    """Accept a cache_id (looked up under cache/task_rows/) or an absolute/relative path."""
    p = Path(spec)
    return str(p) if p.exists() else str(CACHE_ROOT / spec)


def _cache_tasks(cache_path: str):
    """Distinct task names in a TaskRow cache (drives per-task mode: one arm per task)."""
    import glob, pyarrow.parquet as pq
    ts = set()
    for f in glob.glob(f"{cache_path}/data/*.parquet"):
        ts |= set(pq.read_table(f, columns=["task"]).column("task").to_pylist())
    return sorted(ts)


def collection_score(path, legs=LEGS):
    """Per-leg %NLL-reduction + global mean for one COLLECTION run json (the single pooled arm)."""
    d = json.loads(Path(path).read_text())
    base, tasks = d["baseline"], d["tasks"]
    arm = tasks.get("__COLLECTION__") or next(iter(tasks.values()))  # the pooled arm
    per = {}
    for l in legs:
        dk, bk = l + "_delta", l + "_nll"
        if arm.get(dk) is not None and base.get(bk):
            per[l] = -arm[dk] / base[bk] * 100.0
    per["global"] = mean(per.values()) if per else float("nan")
    per["reward_final"] = arm.get("reward_final")
    return per


def out_file(name, run_tag, seed, steps, mix, main, init):
    rt = ("_" + run_tag.lstrip("_")) if run_tag else ""
    return RESULTS / f"influence_COLL-{name}{rt}_S{seed}_T{steps}_M{int(mix*100)}_{main}_{init}.json"


def run_one(name, cache, model, run_tag, main, seed, steps, mix, extra_env, share_baseline=False, per_task=False):
    cache_path = _resolve_cache(cache)
    env = dict(os.environ)
    env.update({
        "AUX_DATASET": "rc", "TASKROW_CACHE": cache_path,
        "MODEL": model, "RUN_TAG": run_tag, "MAIN_DATA": main, "SEED": str(seed),
        "TRAIN_STEPS": str(steps), "MIX_AUX": str(mix),
        "MAIN_LOCAL": env.get("MAIN_LOCAL", "data_cache"), "COMPLETION_ONLY": "1",
        # cloze + letter mmlu, mbpp — same legs as the frozen roster / HH matrix
        "EVAL_MBPP": "1", "EVAL_MMLU_MATH": "1", "EVAL_MMLU_LOGIC": "1",
        "EVAL_MMLU_MATH_CLOZE": "1", "EVAL_MMLU_LOGIC_CLOZE": "1",
        "EVAL_GSM8K": "1", "EVAL_DROP": "1",   # free-gen math + discrete-reasoning legs (nll + exact-match)
        "LOG_SAT": "1" if per_task else "0", "LOG_REWARD": "1", "REWARD_MODE": "instruct",
    })
    # LOG-LARGE default for COLLECTION runs: full held-out + retention + per-subject panel. All cheap
    # answer-NLL passes (~5k extra rows ≈ +1min/cell), so log everything and filter offline; --env still
    # overrides any leg. Per-task mode (172 arms) stays lean — caller slims via --env, heavy panel skipped.
    if not per_task:
        env.setdefault("EVAL_BBH_TEST", "1")          # held-out algorithmic BBH TEST (dev/test hygiene)
        env.setdefault("EVAL_MMLU_OTHER_CLOZE", "1")  # 47-subject general-knowledge retention guardrail
        env.setdefault("EVAL_FOLIO", "1")             # LOGIC: first-order-logic NL entailment, cloze
        env.setdefault("EVAL_FLAN", "1")              # instruction-following capability leg
        env.setdefault("EVAL_MMLU_MATH_MACRO_CLOZE", "1")  # TRUE math macro (5 subj) — plain mmlu_math_cloze=hs_math only
        env.setdefault("EVAL_MMLU_MATH_MACRO", "1")
        for _s in MMLU_SUBJECTS:                       # per-subject MMLU math/logic, BOTH cloze + letter
            env.setdefault(f"EVAL_MMLU_{_s.upper()}", "1")
            env.setdefault(f"EVAL_MMLU_{_s.upper()}_CLOZE", "1")
    # per-task mode: measure EACH task in the cache as its own aux arm (no COLLECTION pooling). Needed for
    # the task-level feature analysis (answer/prompt length, headroom, calibration → influence). LOG_SAT on
    # (saturation = headroom); caller slims heavy eval legs via --env. RUN_TAG carries the collection so the
    # 4 per-task files don't collide (filename lacks a coll field outside COLLECTION mode).
    if per_task:
        env["TASKS"] = ",".join(_cache_tasks(cache_path)); env.pop("COLLECTION", None)
    else:
        env["COLLECTION"] = name
    if share_baseline:  # baseline (main-only) is collection-independent → share across collections/mixes
        slug = model.replace("/", "-")
        env["BASELINE_CACHE"] = str(RESULTS / f"_baseline_{slug}_{main}_S{seed}_T{steps}.json")
    for kv in extra_env:
        k, _, v = kv.partition("="); env[k] = v
    print(f"[coll] {name:<10s} cache={cache} model={model} tag={run_tag}", flush=True)
    subprocess.run([sys.executable, ENGINE], cwd=str(REPO), env=env, check=True)


def cmd_run(a):
    cols = [c for c in a.collections.split(",") if c]
    for c in cols:
        name, _, cache = c.partition("=")
        rt = f"{a.run_tag}{name.upper()}" if a.per_task else a.run_tag   # per-collection tag so per-task files differ
        run_one(name, cache, a.model, rt, a.main, a.seed, a.steps, a.mix, a.env,
                share_baseline=a.share_baseline, per_task=a.per_task)
    if not a.per_task:            # per-task output feeds the offline feature analysis, not the collection table
        cmd_summarize(a)


def cmd_summarize(a):
    rt = ("_" + a.run_tag.lstrip("_")) if a.run_tag else ""
    pat = f"influence_COLL-*{rt}_S{a.seed}_T{a.steps}_M{int(a.mix*100)}_{a.main}_*.json"
    rows = []
    for f in sorted(RESULTS.glob(pat)):
        name = f.name.split("COLL-")[1].split(rt + "_S")[0] if rt else f.name.split("COLL-")[1].split("_S")[0]
        rows.append((name, collection_score(f)))
    if not rows:
        print(f"no COLLECTION results match {pat} in {RESULTS}"); return
    rows.sort(key=lambda r: -r[1]["global"])
    hdr = ["collection", "global"] + LEGS + ["reward"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for name, s in rows:
        cells = [name, f"{s['global']:+.2f}"] + [f"{s[l]:+.2f}" if l in s else "—" for l in LEGS] + \
                [f"{s['reward_final']:.2f}" if s.get('reward_final') is not None else "—"]
        print("| " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "summarize"])
    ap.add_argument("--collections", default="", help="name=cache_id,name=cache_id,… (run only)")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--run-tag", dest="run_tag", default="")
    ap.add_argument("--main", default="fwdolci")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--mix", type=float, default=0.2)
    ap.add_argument("--env", nargs="*", default=[], help="extra K=V passed to the engine (LR, BATCH, …)")
    ap.add_argument("--share-baseline", dest="share_baseline", action="store_true",
                    help="reuse one main-only baseline across all collections at this (model,main,seed,steps)")
    ap.add_argument("--per-task", dest="per_task", action="store_true",
                    help="measure EACH task in every cache as its own aux arm (task-level feature analysis) "
                         "instead of one pooled collection arm; writes influence_<tag><COLL>_... per collection")
    a = ap.parse_args()
    (cmd_run if a.cmd == "run" else cmd_summarize)(a)


if __name__ == "__main__":
    main()
