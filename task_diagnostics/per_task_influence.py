#!/usr/bin/env python3
"""
per_task_influence.py — controlled per-task mixed-data ablation (generic).

For each RC task X, measure the *marginal* effect of mixing it into a main
training stream — one task at a time, fixed baseline, direct loss measurement,
no regression fitting on noisy windowed signals.

Setup (parameterized):
  MAIN_DATA    fw | dolci          (which corpus dominates training)
  FROM_SCRATCH 0 = pretrained      (correct for dolci; FW is in pretraining)
               1 = random init     (correct for fw; matches run_sft.py default)
  Baseline:    train on MAIN_DATA only, TRAIN_STEPS steps (packed full-LM loss)
  Per-task:    interleave(MAIN_DATA 1-MIX_AUX, RC_task_X MIX_AUX), same config
  Compare:     Δ NLL = with_X − baseline_main_only on held-out BBH/Dolci/FW

Env vars:
  SEED        (default 43)
  TRAIN_STEPS (default 500)
  MIX_AUX     (default 0.2)
  MAIN_DATA   (default dolci)         fw | dolci
  FROM_SCRATCH (default 0 if dolci else 1)
  LR          (default 1e-4)
  TASKS       comma-separated subset (default all 40)
"""
import os, sys, json, datetime, logging, warnings, time
from pathlib import Path
import torch

sys.stdout.reconfigure(line_buffering=True)
HF_CACHE = os.environ.get("HF_CACHE", str(Path.home() / "tmp/hf_cache"))
os.environ.update({"HF_HOME": HF_CACHE, "HF_DATASETS_CACHE": HF_CACHE,
                   "TOKENIZERS_PARALLELISM": "false", "WANDB_DISABLED": "true"})
warnings.filterwarnings("ignore")
logging.getLogger("trl").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from datasets import load_dataset, interleave_datasets, IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

ALL_TASKS = [
    'arithmetics', 'bayesian_association', 'bayesian_intervention', 'code_execution',
    'conjecture_entailment', 'constrained_continuation', 'constraint_satisfaction',
    'continuation', 'coreference', 'count_elements', 'diff_prediction',
    'equation_system', 'evidence_retrieval', 'graph_dependencies', 'graph_isomorphism',
    'graph_pathfinding', 'graph_successors', 'lambda_reduction', 'lexical_knowledge',
    'locate_error', 'logic_formalization', 'logic_nli', 'navigation', 'parsability',
    'parsing', 'planning', 'proof_reconstruction', 'qualitative_reasoning',
    'reference_tracking', 'regex_following', 'regex_induction', 'regex_reasoning',
    'sequential_induction', 'set_equality', 'set_intersection', 'set_missing_element',
    'table_conversion', 'table_qa',
]   # NB: diff_patching + term_unification REMOVED from the rc dataset on 2026-06-04 — dropped
    # here so a streamed run doesn't scan the whole pile forever hunting a non-existent task.
_DEFAULT_RC_TASKS = list(ALL_TASKS)   # full pool for PEER sampling (survives a TASKS filter)

SEED         = int(os.environ.get("SEED", 43))
TRAIN_STEPS  = int(os.environ.get("TRAIN_STEPS", 500))
MIX_AUX      = float(os.environ.get("MIX_AUX", 0.2))
# TOKEN_RATIO: interleave_datasets samples per EXAMPLE, so a fixed example-prob (MIX_AUX) gives each
# collection an aux (prompt+answer) TOKEN slice ∝ its mean example length → verbose collections eat a
# bigger share of the fixed packed-compute budget. When set, reinterpret MIX_AUX as the TARGET aux
# TOKEN fraction and solve for the example-prob that hits it (measured mean lengths). Equalizes the
# aux compute-slice across collections; reduces to example-matching when L_aux==L_main. COLLECTION arm
# only. Off by default (backward-compatible). NB: matches total tokens, NOT answer tokens (completion-only).
TOKEN_RATIO  = os.environ.get("TOKEN_RATIO", "0") != "0"
# OVERSAMPLE: measure MARGINAL value of upweighting a task IN a full aux mixture.
#   baseline = main(1-MIX) + all-aux(MIX);  treatment_X = main(1-MIX) + X(MIX/2) + all-aux(MIX/2)
#   delta = NLL(treatment_X) - NLL(baseline_mix). Total aux stays MIX (fair compute).
OVERSAMPLE   = os.environ.get("OVERSAMPLE", "0") != "0"
# OVS_X_FRAC: fraction of the aux budget (MIX) given to the measured task X; the rest goes to the
# all-rc background. Default 0.5 (the classic 50/50 split). Small values (e.g. 0.1) = the REALISTIC
# marginal setting: a large all-rc mixture + a small extra dose of X (X is already ~1/N in the mix).
OVS_X_FRAC   = float(os.environ.get("OVS_X_FRAC", "0.5"))
# PEER_MIX: like OVERSAMPLE but the aux background is a SMALL fixed sample of N_PEERS
#   peer tasks (not the full pile) — a more realistic "this task among a few others"
#   mixture. baseline = main(1-MIX) + peers(MIX) [trained once, peer set fixed by SEED];
#   treatment_X = main(1-MIX) + X(MIX/2) + peers(MIX/2); delta isolates upweighting X
#   on top of a realistic K-peer background. Default off.
PEER_MIX     = os.environ.get("PEER_MIX", "0") != "0"
N_PEERS      = int(os.environ.get("N_PEERS", 8))
# COMPLETION_ONLY: mask the prompt in the loss (answer-only), MATCHING run_sft.py
# production (completion_only_loss=True). Default ON. Set 0 for legacy full-LM loss.
COMPLETION_ONLY = os.environ.get("COMPLETION_ONLY", "1") != "0"
# MODE_FILTER: restrict the per-task aux stream to one answer style from the rc `mode`
# column (instruct | cot | few_shot | verification). Empty = pool all modes (canonical).
# cot answers embed the reasoning trace (longer/structured target); instruct emits only the
# final answer. Lets us isolate whether cot vs instruct examples help or hurt.
MODE_FILTER  = os.environ.get("MODE_FILTER", "").strip()
# LEVEL_MAX: cap the per-task aux stream (train + sat/reward eval) to instances with level <= LEVEL_MAX
# ("easier calibration"). Tasks calibrated too hard sit past the inverted-U peak and hurt; capping tests
# them where they are learnable, and matches rc/rg to a common easy calibration. Empty = all levels.
LEVEL_MAX    = os.environ.get("LEVEL_MAX", "").strip()
def _level_ok(x):
    if not LEVEL_MAX: return True
    try: return int(x.get("level")) <= int(LEVEL_MAX)
    except (TypeError, ValueError): return True   # rows without a numeric level pass through
# MODE_MIX: realistic FULL-MIXTURE answer-format ablation — aux = ALL tasks, but the mode column
# blended at given weights (e.g. "verification:0.5,instruct:0.5"). Runs ONE pseudo-task __ALLMIX__
# (main + the blended all-aux stream at MIX_AUX) vs the main-only baseline. Empty = off.
MODE_MIX     = os.environ.get("MODE_MIX", "").strip()
_mode_mix    = []
if MODE_MIX:
    for _p in MODE_MIX.split(","):
        _m, _w = _p.split(":"); _mode_mix.append((_m.strip(), float(_w)))
    _ws = sum(w for _, w in _mode_mix); _mode_mix = [(m, w/_ws) for m, w in _mode_mix]
MAIN_DATA    = os.environ.get("MAIN_DATA", "dolci").lower()
assert MAIN_DATA in ("fw", "dolci", "flan", "fw_recent", "tasksource", "fwdolci", "fwtasksource", "codealpaca", "fwdolcicode"), MAIN_DATA
# Default to the local cache when present so EVERY run is stream-free (HF xet-bridge 408s crash long
# streaming runs). Set MAIN_LOCAL="" to force HF streaming. Training data = same first-20k docs as the
# stream; only the eval slice differs (skip 25k) → BBH (primary) identical, dolci/fw deltas shift slightly.
MAIN_LOCAL   = os.environ.get("MAIN_LOCAL", "data_cache" if Path("data_cache/fw_main.jsonl").exists() else "")
def _read_jsonl(p):
    return [json.loads(l) for l in open(p)]
_default_scratch = "1" if MAIN_DATA == "fw" else "0"
FROM_SCRATCH = os.environ.get("FROM_SCRATCH", _default_scratch) != "0"
LR           = float(os.environ.get("LR", 1e-4))
MAX_LEN      = int(os.environ.get("MAX_LEN", "512"))   # paper protocol sets MAX_LEN=1024 (cuts SynLogic discard ~0)
BATCH        = int(os.environ.get("BATCH", 8))   # reduce for larger decoders (e.g. BATCH=2 for 400-600M)
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM", 1))  # effective batch = BATCH*GRAD_ACCUM; raise for cleaner PRETRAINING-regime gradients (max_steps counts optimizer steps → N× more tokens)

# Aux-task source label for result directories/task lists. Aux rows themselves come from TASKROW_CACHE.
AUX_DATASET  = os.environ.get("AUX_DATASET", "rc").lower()
assert AUX_DATASET in ("rc", "rgym", "basic"), AUX_DATASET
_RESULTS_SUB = {"rc": "per_task_results", "rgym": "per_task_results_rgym",
                "basic": "per_task_results_bproc"}[AUX_DATASET]
# non-rc datasets carry their own task list. rgym list lives in the reasoning_gym task module
# (RGYM_TASKS); basic still reads basic_tasks.json in cwd; rc uses the builtin default list.
def _load_tasklist(ds):
    if ds == "rgym":
        from reasoning_core.tasks._reasoning_gym import RGYM_TASKS
        return list(RGYM_TASKS)
    if ds == "basic":
        return json.loads(Path("basic_tasks.json").read_text())["tasks"]
    return None
if AUX_DATASET in ("rgym", "basic"):
    ALL_TASKS = _load_tasklist(AUX_DATASET)
_filter      = os.environ.get("TASKS", "").strip()
if _filter: ALL_TASKS = [t.strip() for t in _filter.split(",") if t.strip()]
if MODE_MIX: ALL_TASKS = ["__ALLMIX__"]     # single full-mixture arm under the mode blend
# GROUP mode: measure a CHOSEN SET of tasks POOLED into ONE aux arm at the SAME total MIX budget
# (round-robin so each member gets MIX/|group| dose), to test complementarity against the individual
# per-task deltas. e.g. GROUP_TASKS=logic_nli,multistep_nli  → one arm, comparable to each alone.
GROUP_TASKS = [t.strip() for t in os.environ.get("GROUP_TASKS", "").split(",") if t.strip()]
if GROUP_TASKS: ALL_TASKS = ["__GROUP__"]
# COLLECTION mode: score a whole NAMED collection (rc / rgym / pw / synlogic) as ONE pooled aux arm —
# the deployment-realistic "train on all of it at MIX_AUX" number. It is comparable ACROSS collections
# of different task counts (the per-task MEAN is confounded by how many/which tasks a collection has),
# so it's the right figure for the paper's collection table. Reuses all_aux_gen (every task in the
# TASKROW_CACHE) as the single X arm vs the main-only baseline. COLLECTION=<name> is just the label.
COLLECTION = os.environ.get("COLLECTION", "").strip()
if COLLECTION: ALL_TASKS = ["__COLLECTION__"]
# PEER_MIX background: fixed seeded sample of N_PEERS tasks (independent of target).
import random as _random
_FULL_TASKS  = (_load_tasklist(AUX_DATASET)
                if AUX_DATASET in ("rgym", "basic") else list(_DEFAULT_RC_TASKS))
PEER_TASKS   = sorted(_random.Random(SEED).sample(
                   _FULL_TASKS, min(N_PEERS, len(_FULL_TASKS)))) if PEER_MIX else []

MODEL_NAME = os.environ.get("MODEL", "HuggingFaceTB/SmolLM2-135M")  # e.g. jhu-clsp/ettin-decoder-400m, Qwen/Qwen3-0.6B (use a distinct RUN_TAG)
# WARMED_CKPT_DIR is only used for the fw-pretraining warm-start baseline; env-overridable.
WARMED_CKPT_DIR = Path(os.environ.get("WARMED_CKPT_DIR",
                      "/mnt/nfs_share_magnet2/dsileo/sandboxes/rc_grad/model_checkpoints/"
                      "HuggingFaceTB_SmolLM2-135M_fw_rc_W300"))
# OUT_DIR: where raw result JSONs are written. Default = <repo>/per_task_results (this file
# lives in <repo>/task_diagnostics/). The orchestrator passes OUT_DIR explicitly so the two always agree.
OUT_DIR     = Path(os.environ.get("OUT_DIR") or (Path(__file__).resolve().parent.parent / _RESULTS_SUB))
OUT_DIR.mkdir(parents=True, exist_ok=True)
_init_tag   = "scratch" if FROM_SCRATCH else "pretrained"
_ovs_tag    = ("_OVS" + (f"x{int(OVS_X_FRAC*100)}" if OVS_X_FRAC != 0.5 else "")) if OVERSAMPLE else ""
_peer_tag   = f"_PEERS{N_PEERS}" if PEER_MIX else ""
_co_tag     = "" if COMPLETION_ONLY else "_FULLLM"   # answer-only = canonical clean name
_mode_tag   = f"_MODE-{MODE_FILTER}" if MODE_FILTER else ""
_level_tag  = f"_L{LEVEL_MAX}" if LEVEL_MAX else ""                       # easier-calibration runs → distinct file
_grp_tag    = ("_GRP-" + "+".join(GROUP_TASKS)) if GROUP_TASKS else ""   # distinct file per group
_coll_tag   = ("_COLL-" + COLLECTION) if COLLECTION else ""              # clean name (not 50 task names)
_tr_tag     = "_TR" if TOKEN_RATIO else ""                               # token-matched cells → distinct files
# RUN_TAG: optional free-form suffix to write to distinct files without overwriting canonical
# results (e.g. re-measuring tasks after the rc dataset was updated on HF). Default OFF.
_run_tag    = os.environ.get("RUN_TAG", "")
if _run_tag and not _run_tag.startswith("_"): _run_tag = "_" + _run_tag
_tag        = f"{_ovs_tag}{_peer_tag}{_co_tag}{_mode_tag}{_level_tag}{_grp_tag}{_coll_tag}{_tr_tag}{_run_tag}"
OUT_FILE    = OUT_DIR / f"influence{_tag}_S{SEED}_T{TRAIN_STEPS}_M{int(MIX_AUX*100)}_{MAIN_DATA}_{_init_tag}.json"
LOG_PER_EX  = os.environ.get("LOG_PER_EX", "0") != "0"
PEREX_FILE  = OUT_DIR / f"perex{_tag}_S{SEED}_T{TRAIN_STEPS}_M{int(MIX_AUX*100)}_{MAIN_DATA}_{_init_tag}.json"
# LOG_SAT: record per-task answer-token-accuracy curve DURING the treatment run
# (in-mixture saturation — measured while the 80% main/FW background is present, which
# is the realistic difficulty signal we want; supersedes the standalone profiler).
# Default ON: merged into the influence run so we don't retrain each task twice.
LOG_SAT     = os.environ.get("LOG_SAT", "1") != "0"
SAT_EVERY   = int(os.environ.get("SAT_EVERY", 50))
SAT_NEVAL   = int(os.environ.get("SAT_NEVAL", 40))
# LOG_REWARD: also log FREE-GENERATION exact-match reward at BEGIN + END of training. This is the
# honest learnability signal — teacher-forced token_acc is an inverted proxy (see analogical: tf
# 0.63→0.80 but free-gen ~0.04). Reuses the SAME held-out sat rows (no fresh data). Default ON.
LOG_REWARD   = os.environ.get("LOG_REWARD", "1") != "0"
REWARD_NEVAL = int(os.environ.get("REWARD_NEVAL", 25))
REWARD_MAXTOK= int(os.environ.get("REWARD_MAXTOK", 256))
# REWARD_MODE: gates ONLY the free-gen score_answer reward (NOT token_acc/sat, which is a next-token
# measure valid on every mode). score_answer on the pile's mode blend conflates generation solvability
# with trivial verification Yes/No, so restrict the reward to a single clean mode. Default "instruct"
# (the real free-gen task). "" / "all" = no filter.
REWARD_MODE  = os.environ.get("REWARD_MODE", "instruct")
SAT_FILE    = OUT_DIR / f"sat{_ovs_tag}{_peer_tag}{_mode_tag}{_tr_tag}{_run_tag}_S{SEED}_T{TRAIN_STEPS}_M{int(MIX_AUX*100)}_{MAIN_DATA}_{_init_tag}.json"
# CKPT_EVAL: budget-convergence study — eval BBH/Dolci/FW every CKPT_EVAL steps DURING training
# (baseline + each task), so influence@T = task_nll@T - baseline_nll@T from ONE run per task. 0=off.
CKPT_EVAL   = int(os.environ.get("CKPT_EVAL", "0"))
CKPT_FILE   = OUT_DIR / f"ckpt{_tr_tag}{_run_tag}_S{SEED}_T{TRAIN_STEPS}_M{int(MIX_AUX*100)}_{MAIN_DATA}_{_init_tag}.json"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

torch.manual_seed(SEED)
_mode_str = ("PEER_MIX(%d peers)" % N_PEERS if PEER_MIX else
             "OVERSAMPLE" if OVERSAMPLE else "single-task")
print(f"\n=== Controlled ablation — seed={SEED}  steps={TRAIN_STEPS}  "
      f"mix={1-MIX_AUX:.0%} {MAIN_DATA} + {MIX_AUX:.0%} aux | init={_init_tag} | "
      f"mode={_mode_str} ===\n")
if PEER_MIX: print(f"  peer background ({N_PEERS}): {PEER_TASKS}\n")

# ── Tokenizer + model + init snapshot ───────────────────────────────────────────
try:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
except Exception as _tok_e:
    # Some recent models (e.g. LiquidAI LFM2.5, tokenizer_class="TokenizersBackend") are not in this
    # transformers version's tokenizer registry. Load the fast tokenizer straight from tokenizer.json
    # and restore special tokens from tokenizer_config.json.
    from transformers import PreTrainedTokenizerFast
    from huggingface_hub import hf_hub_download
    _tj = hf_hub_download(MODEL_NAME, "tokenizer.json")
    _tc = json.loads(Path(hf_hub_download(MODEL_NAME, "tokenizer_config.json")).read_text())
    _val = lambda v: (v.get("content") if isinstance(v, dict) else v)
    _specials = {k: _val(_tc[k]) for k in ("bos_token", "eos_token", "unk_token", "pad_token") if _tc.get(k)}
    tok = PreTrainedTokenizerFast(tokenizer_file=_tj, **_specials)
    print(f"⚠️ Fallback PreTrainedTokenizerFast for {MODEL_NAME} (specials={list(_specials)}); reason: {repr(_tok_e)[:80]}")
if tok.pad_token is None: tok.pad_token = tok.eos_token
EOS = tok.eos_token

if FROM_SCRATCH:
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_config(cfg, dtype=DTYPE,
                                              attn_implementation="sdpa").to(DEVICE)
    print(f"🆕 Random-init {MODEL_NAME} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
elif MAIN_DATA in ("dolci", "flan", "tasksource", "fw_recent", "fwdolci", "fwtasksource", "codealpaca", "fwdolcicode"):
    # Use raw pretrained model. dolci/flan/tasksource = instruction FT (novel data);
    # fw_recent = continued pretraining on FRESH post-cutoff FineWeb; fwdolci/fwtasksource = realistic
    # BLEND of continued-pretraining text + instruction (mid-training regime). All start from pretrained.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, attn_implementation="sdpa",
    ).to(DEVICE)
    print(f"📦 Pretrained {MODEL_NAME}")
else:
    # MAIN_DATA=fw + pretrained: use cached FW-warmed checkpoint for fair baseline
    model = AutoModelForCausalLM.from_pretrained(
        WARMED_CKPT_DIR, dtype=DTYPE, attn_implementation="sdpa",
    ).to(DEVICE)
    print(f"♻️  FW-warmed checkpoint from {WARMED_CKPT_DIR.name}")

INIT_STATE = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
def reset_model():
    model.load_state_dict({k: v.to(DEVICE) for k, v in INIT_STATE.items()})

# ── Held-out evals (BBH, Dolci, FW) — eval all three regardless of main_data ──
# ── BBH DEV/TEST split (held-out discipline on the EVAL itself) ──────────────────────────────────
# DEV = 12 BBH-NLP tasks: the ONLY BBH used to revise generators, tune difficulty, select tasks, and
# choose mixture weights. TEST = 11 BBH-algorithmic tasks: touched ONLY for the final headline number
# (gated EVAL_BBH_TEST=1) so selection can't overfit what we report. The `bbh` leg (selection) = DEV.
BBH_DEV = [
    "causal_judgement", "date_understanding", "disambiguation_qa", "formal_fallacies",
    "hyperbaton", "movie_recommendation", "penguins_in_a_table",
    "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection",
    "snarks", "sports_understanding",
]
BBH_TEST = [   # lukaemon/bbh splits two of these by object count → use a representative variant
    "boolean_expressions", "dyck_languages", "geometric_shapes", "logical_deduction_five_objects",
    "multistep_arithmetic_two", "navigate", "object_counting", "temporal_sequences",
    "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting",
]
BBH_SUBTASKS = BBH_DEV   # back-compat alias; the default `bbh` leg is the DEV set
print("📊 Building eval sets…")
import re as _re
_BBH_BOOL = [("True", "False"), ("Yes", "No"), ("valid", "invalid")]   # fixed-label BBH tasks
def _bbh_meta(inp, target):
    """(choices, gold_idx) for MC max-likelihood scoring of one BBH example; None if free-form.
    CLOZE-consistent with MMLU/FOLIO: choices are the option TEXTs (e.g. 'five', '02/16/2009'), NOT the
    letter labels '(A)' — score answer content, not the label. Letter-option tasks → parse the '(X) text'
    block; boolean/label tasks → the fixed pair (already text). Free-form
    (object_counting/word_sorting/dyck/arithmetic) → None, skipped in MC acc.
    (Unlike MMLU cloze, BBH options stay in the prompt — they ARE the question — so this is text-scoring
    with options visible, the closest cloze form possible for BBH.)"""
    tgt = str(target).replace(EOS, "").strip()
    mt = _re.match(r"\(?([A-Z])\)?$", tgt)                              # multiple-choice letter task
    if mt:
        region = inp.split("Options:")[-1]                             # parse only the options block
        opts = _re.findall(r"\(([A-Z])\)\s*(.+?)\s*(?=\n\s*\([A-Z]\)|\Z)", region, _re.S)
        letters = [L for L, _ in opts]; texts = [t.strip() for _, t in opts]
        if len(letters) >= 2 and mt.group(1) in letters and all(texts):
            return (texts, letters.index(mt.group(1)))               # score option TEXT (cloze)
        return None
    for pair in _BBH_BOOL:                                              # boolean / fixed-label task
        low = [x.lower() for x in pair]
        if tgt.lower() in low: return (list(pair), low.index(tgt.lower()))
    return None
def _build_bbh(subtasks):
    ev, meta = [], []
    for sub in subtasks:
        try:
            for ex in list(load_dataset("lukaemon/bbh", sub, split="test"))[5:25]:
                ev.append((f"{ex['input']}\n", f"{ex['target']}{EOS}"))
                meta.append(_bbh_meta(ex["input"], ex["target"]))
        except Exception as e: print(f"  ⚠ {sub}: {e}")
    return ev, meta
BBH_EVAL, BBH_META = _build_bbh(BBH_DEV)
EVAL_BBH_TEST = os.environ.get("EVAL_BBH_TEST", "0") == "1"   # final, held-out algorithmic TEST eval
BBH_TEST_EVAL, BBH_TEST_META = _build_bbh(BBH_TEST) if EVAL_BBH_TEST else ([], [])
# `bbh_open` leg: options-OMITTED BBH cloze over DEV subtasks whose answer is generable (number/date/name/
# category/binary) — strip the menu, score the gold answer TEXT as a free continuation (answer NOT in prompt,
# like MMLU cloze) → a BBH reasoning signal with zero MCQ-format confound. Comparison-bound tasks
# (snarks/movie_recommendation/ruin_names/hyperbaton) are excluded (they need the candidate list).
BBH_OPEN_SUBS = ["reasoning_about_colored_objects", "penguins_in_a_table", "date_understanding",
                 "disambiguation_qa", "salient_translation_error_detection", "causal_judgement",
                 "formal_fallacies", "sports_understanding"]
EVAL_BBH_OPEN = os.environ.get("EVAL_BBH_OPEN", "1") == "1"
def _build_bbh_open(subs):
    ev = []
    for sub in subs:
        try:
            for ex in list(load_dataset("lukaemon/bbh", sub, split="test"))[5:25]:
                m = _bbh_meta(ex["input"], ex["target"])                          # letter/bool → gold option TEXT
                gold = m[0][m[1]] if m else str(ex["target"]).replace(EOS, "").strip()   # else raw target (free-form)
                if not m and _re.match(r"\(?[A-Z]\)?$", gold): continue           # unparsed letter-MCQ → skip (no bare letter)
                ev.append((f"{ex['input'].split('Options:')[0].strip()}\nAnswer:", f"{gold}{EOS}"))
        except Exception as e: print(f"  ⚠ open {sub}: {e}")
    return ev
BBH_OPEN_EVAL = _build_bbh_open(BBH_OPEN_SUBS) if EVAL_BBH_OPEN else []
# same treatment for the held-out TEST set (bbh_test_open): generable TEST subtasks — free-form
# (arithmetic/word_sorting/dyck/object_counting → raw numeric/text target) + binary (boolean_expr/navigate/
# web_of_lies) + letter-but-nameable (geometric_shapes/temporal/tracking). logical_deduction excluded (comparison-bound).
BBH_TEST_OPEN_SUBS = ["boolean_expressions", "navigate", "web_of_lies", "object_counting",
    "multistep_arithmetic_two", "word_sorting", "dyck_languages", "geometric_shapes",
    "temporal_sequences", "tracking_shuffled_objects_three_objects"]
BBH_TEST_OPEN_EVAL = _build_bbh_open(BBH_TEST_OPEN_SUBS) if (EVAL_BBH_OPEN and EVAL_BBH_TEST) else []
DOLCI_EVAL, FW_EVAL = [], []
if MAIN_LOCAL:                                   # stream-free eval slices
    for r in _read_jsonl(Path(MAIN_LOCAL) / "dolci_eval.jsonl"):
        if r.get("answer"): DOLCI_EVAL.append((f"{r['prompt']}\n", f"{r['answer']}{EOS}"))
    _fw_eval_name = "fw_recent_eval.jsonl" if MAIN_DATA == "fw_recent" else "fw_eval.jsonl"
    for r in _read_jsonl(Path(MAIN_LOCAL) / _fw_eval_name):
        txt = r["text"][:1500]
        if len(txt) >= 100: FW_EVAL.append(txt)
    DOLCI_EVAL, FW_EVAL = DOLCI_EVAL[:200], FW_EVAL[:200]
else:
    for x in load_dataset("tasksource/dolci-instruct", split="train", streaming=True).skip(50_000):
        if not x.get("answer"): continue
        DOLCI_EVAL.append((f"{x['prompt']}\n", f"{x['answer']}{EOS}"))
        if len(DOLCI_EVAL) >= 200: break
    for x in load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True).skip(800_000):
        txt = x["text"][:1500]
        if len(txt) < 100: continue
        FW_EVAL.append(txt)
        if len(FW_EVAL) >= 200: break

# Optional held-out TRANSFER targets beyond the always-on BBH/Dolci/FW. Each is a QA-NLL (answer-only)
# eval gated by EVAL_<NAME>=1, loaded from a {prompt,answer} jsonl. Adding a capability leg = one line
# in _EXTRA_SPEC. Output schema per task: <name>_nll / <name>_delta. Capability map for the aggregate:
#   fw=WEB_LM, bbh=REASONING, mmlu_math=MATH, mmlu_logic=LOGIC, mbpp=CODING, dolci=CONVERSATION.
_EXTRA_SPEC = [   # (name, default_jsonl, cap)
    ("flan",       "data_cache/flan_eval.jsonl",       200),
    ("mbpp",       "data_cache/mbpp_eval.jsonl",       257),
    ("mmlu_math",  "data_cache/mmlu_math_eval.jsonl",  400),
    ("mmlu_logic", "data_cache/mmlu_logic_eval.jsonl", 200),
    ("mmlu_math_cloze",  "data_cache/mmlu_math_cloze_eval.jsonl",  400),   # format-fair: answer-text NLL
    ("mmlu_logic_cloze", "data_cache/mmlu_logic_cloze_eval.jsonl", 200),   # (no listed options in prompt)
    # Per-subject MMLU cloze (build_mmlu_subjects.py). Separate legs so math/logic transfer is read per
    # subject and a broader math aggregate reconstructed offline. Each gated EVAL_MMLU_<SUBJECT>_CLOZE=1.
    ("mmlu_abstract_algebra_cloze",       "data_cache/mmlu_abstract_algebra_cloze_eval.jsonl",       400),
    ("mmlu_college_mathematics_cloze",    "data_cache/mmlu_college_mathematics_cloze_eval.jsonl",    400),
    ("mmlu_elementary_mathematics_cloze", "data_cache/mmlu_elementary_mathematics_cloze_eval.jsonl", 400),
    ("mmlu_high_school_mathematics_cloze","data_cache/mmlu_high_school_mathematics_cloze_eval.jsonl",400),
    ("mmlu_high_school_statistics_cloze", "data_cache/mmlu_high_school_statistics_cloze_eval.jsonl", 400),
    ("mmlu_formal_logic_cloze",           "data_cache/mmlu_formal_logic_cloze_eval.jsonl",           400),
    # normal/letter twins (options in prompt, gold LETTER) for the same 6 subjects — gate EVAL_MMLU_<SUBJECT>=1.
    ("mmlu_abstract_algebra",       "data_cache/mmlu_abstract_algebra_eval.jsonl",       400),
    ("mmlu_college_mathematics",    "data_cache/mmlu_college_mathematics_eval.jsonl",    400),
    ("mmlu_elementary_mathematics", "data_cache/mmlu_elementary_mathematics_eval.jsonl", 400),
    ("mmlu_high_school_mathematics","data_cache/mmlu_high_school_mathematics_eval.jsonl",400),
    ("mmlu_high_school_statistics", "data_cache/mmlu_high_school_statistics_eval.jsonl", 400),
    ("mmlu_formal_logic",           "data_cache/mmlu_formal_logic_eval.jsonl",           400),
    ("folio",            "data_cache/folio_eval.jsonl",            203),   # LOGIC: first-order-logic NL entailment (True/False/Uncertain), cloze — gate EVAL_FOLIO=1
    # Free-gen capability legs (build_gsm8k_drop.py): short gold answer → answer-NLL (*_nll) AND numeric-aware
    # free-gen exact-match (*_em, under EVAL_ACC). Default ON in the collection driver. gate EVAL_GSM8K/EVAL_DROP=1.
    ("gsm8k",            "data_cache/gsm8k_eval.jsonl",            300),   # MATH: grade-school word problems, final-number answer
    ("drop",             "data_cache/drop_eval.jsonl",             300),   # NUM/READING: discrete reasoning over paragraphs, span/number answer
    # RETENTION GUARDRAIL (not a selection leg): MMLU ∖ {math, formal_logic, CS}, 47 subjects × 25, cloze.
    # Per-example NLL is saved (LOG_PER_EX=1) so the subject-macro-average + subject-bootstrap are computed
    # offline; the pooled *_delta here is only a coarse summary. Measures retention of broad academic domains.
    ("mmlu_other_cloze", "data_cache/mmlu_other_cloze_eval.jsonl", 1175),
]
EXTRA_EVALS = {}   # name -> [(prompt, answer)]
EXTRA_META  = {}   # name -> [(choices, gold_idx) | None]   for MCQ choice-scoring accuracy
def _gold_idx(choices, answer):
    """robust gold index: prefer the choice whose text == stored answer; else the dataset's answer_idx."""
    for i, c in enumerate(choices):
        if str(c).strip() == str(answer).strip():
            return i
    return None
for _nm, _path, _cap in _EXTRA_SPEC:
    if os.environ.get(f"EVAL_{_nm.upper()}", "0") == "1":
        _p = os.environ.get(f"{_nm.upper()}_EVAL_PATH", _path)
        if not Path(_p).exists():                 # default-on leg on a machine lacking the jsonl → skip, don't crash
            print(f"  ⚠️  EVAL_{_nm.upper()}=1 but {_p} missing → skipping leg (build it with the matching builder)")
            continue
        _pairs, _meta = [], []
        for r in _read_jsonl(Path(_p)):
            if not (r.get("prompt") and r.get("answer") is not None): continue
            _pairs.append((f"{r['prompt']}\n", f"{r['answer']}{EOS}"))
            _ch = r.get("choices")
            if _ch:
                _gi = _gold_idx(_ch, r["answer"])
                if _gi is None and r.get("answer_idx") is not None: _gi = int(r["answer_idx"])
                _meta.append((_ch, _gi) if _gi is not None else None)
            else:
                _meta.append(None)
        EXTRA_EVALS[_nm] = _pairs[:_cap]
        EXTRA_META[_nm]  = _meta[:_cap]
print(f"  BBH={len(BBH_EVAL)}  Dolci={len(DOLCI_EVAL)}  FW={len(FW_EVAL)}"
      + "".join(f"  {k}={len(v)}" for k, v in EXTRA_EVALS.items()) + "\n")


@torch.no_grad()
def eval_qa(examples):
    model.eval()
    total_loss, total_tok = 0., 0
    per_ex = []
    for prompt, answer in examples:
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        a_ids = tok(answer, add_special_tokens=False).input_ids
        if len(p_ids) + len(a_ids) > MAX_LEN: per_ex.append(None); continue  # keep per_ex index-aligned with `examples` (subject map)
        ids = torch.tensor([p_ids + a_ids], device=DEVICE)
        labels = ids.clone(); labels[0, :len(p_ids)] = -100
        out = model(ids, labels=labels)
        n = (labels[0, 1:] != -100).sum().item()
        total_loss += out.loss.item() * n
        total_tok  += n
        per_ex.append(out.loss.item())
    return total_loss / max(total_tok, 1), total_tok, per_ex

@torch.no_grad()
def eval_lm(texts):
    model.eval()
    total_loss, total_tok = 0., 0
    per_ex = []
    for txt in texts:
        ids = tok(txt, add_special_tokens=False, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").input_ids.to(DEVICE)
        if ids.shape[1] < 2: continue
        out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        total_loss += out.loss.item() * n
        total_tok  += n
        per_ex.append(out.loss.item())
    return total_loss / max(total_tok, 1), total_tok, per_ex

def eval_all():
    bbh_m,   _, bbh_px   = eval_qa(BBH_EVAL)
    dolci_m, _, dolci_px = eval_qa(DOLCI_EVAL)
    fw_m,    _, fw_px    = eval_lm(FW_EVAL)
    perex = {"bbh": bbh_px, "dolci": dolci_px, "fw": fw_px}
    extra = {}
    if BBH_TEST_EVAL:   # held-out algorithmic BBH-TEST — final report only, stored as bbh_test_nll/_delta
        sm, _, spx = eval_qa(BBH_TEST_EVAL)
        extra["bbh_test"] = sm; perex["bbh_test"] = spx
    if BBH_OPEN_EVAL:   # options-omitted BBH cloze (answer generated, no visible menu) → bbh_open_nll/_delta
        om, _, opx = eval_qa(BBH_OPEN_EVAL)
        extra["bbh_open"] = om; perex["bbh_open"] = opx
    if BBH_TEST_OPEN_EVAL:   # options-omitted BBH-TEST cloze → bbh_test_open_nll/_delta
        tom, _, topx = eval_qa(BBH_TEST_OPEN_EVAL)
        extra["bbh_test_open"] = tom; perex["bbh_test_open"] = topx
    for _nm, _ex in EXTRA_EVALS.items():
        m, _, px = eval_qa(_ex)
        extra[_nm] = m
        perex[_nm] = px
    return (bbh_m, dolci_m, fw_m, extra), perex

# ── Held-out ACCURACY (gated EVAL_ACC=1) — reviewers ask for accuracy, not just NLL. ─────────────
# MCQ legs (mmlu_*_cloze): choice-scoring — predict argmin length-normalized NLL over the candidate
# answer TEXTS; accuracy = pred == gold. BBH: greedy-decode exact-match (normalized) vs the gold string.
EVAL_ACC = os.environ.get("EVAL_ACC", "1") == "1"   # default ON: MCQ-cloze acc is free (choice-scoring), BBH exact-match is capped 24-tok greedy (~few min) — cheap enough to always log; set EVAL_ACC=0 to skip

@torch.no_grad()
def _cand_nll(p_ids, cand):
    """length-normalized NLL of candidate answer text `cand` given tokenized prompt ids p_ids."""
    a_ids = tok(str(cand), add_special_tokens=False).input_ids
    if not a_ids or len(p_ids) + len(a_ids) > MAX_LEN: return float("inf")
    ids = torch.tensor([p_ids + a_ids], device=DEVICE)
    labels = ids.clone(); labels[0, :len(p_ids)] = -100
    return model(ids, labels=labels).loss.item()          # mean over answer tokens = length-normalized

@torch.no_grad()
def eval_mcq_acc(examples, metas):
    """choice-scoring accuracy + mean length-normalized NLL of the GOLD answer TEXT (content/cloze NLL),
    over MCQ candidates; metas[i]=(choices, gold_idx)|None. Returns (acc, total, gold_content_nll)."""
    model.eval(); correct = total = 0; gnll = 0.0; gn = 0
    for (prompt, _a), meta in zip(examples, metas):
        if not meta: continue
        choices, gold = meta
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        scores = [_cand_nll(p_ids, c) for c in choices]
        if all(s == float("inf") for s in scores): continue
        pred = min(range(len(scores)), key=lambda i: scores[i])
        correct += int(pred == gold); total += 1
        if scores[gold] != float("inf"): gnll += scores[gold]; gn += 1
    return ((correct / total) if total else None, total, (gnll / gn) if gn else None)

@torch.no_grad()
def eval_gen_acc(examples, max_new=24):
    """greedy-decode normalized exact-match accuracy (BBH: mixed MCQ/free-form, gold string stored)."""
    import re
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    model.eval(); correct = total = 0
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for prompt, answer in examples:
        gold = answer.replace(EOS, "").strip()
        p_ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)
        if p_ids.shape[1] >= MAX_LEN: continue
        g_ids = tok(gold, add_special_tokens=False).input_ids
        gen = model.generate(p_ids, max_new_tokens=min(max_new, len(g_ids) + 6),
                             do_sample=False, pad_token_id=pad)
        out = tok.decode(gen[0][p_ids.shape[1]:], skip_special_tokens=True).split("\n")[0]
        correct += int(norm(out) == norm(gold)); total += 1
    return (correct / total, total) if total else (None, 0)

# Free-gen EXACT-MATCH legs (no MCQ choices): greedy-decode + answer extraction. Numeric-aware — if the
# gold parses as a number (GSM8K, many DROP) compare the LAST number in the generation (tolerates a short
# chain); else normalized-string match the first line (DROP spans). Cheap: GEN_EM_MAXNEW tokens, GEN_EM_N rows.
_GEN_EM_LEGS   = {"gsm8k", "drop"}
GEN_EM_MAXNEW  = int(os.environ.get("GEN_EM_MAXNEW", 64))
GEN_EM_N       = int(os.environ.get("GEN_EM_N", 200))
_NUM_RE        = _re.compile(r"-?\d[\d,]*(?:\.\d+)?")
def _last_num(s):
    m = _NUM_RE.findall(s)
    return m[-1].replace(",", "") if m else None

def _tok_f1(pred, gold):
    """SQuAD/DROP-style token-overlap F1: lowercase, drop articles+punctuation, whitespace-tokenize, bag-F1."""
    def _norm(s):
        s = _re.sub(r"\b(a|an|the)\b", " ", s.lower())
        return _re.sub(r"[^a-z0-9 ]", " ", s).split()
    p, g = _norm(pred), _norm(gold)
    if not p or not g: return float(p == g)                    # both empty → 1.0; one empty → 0.0
    from collections import Counter
    common = sum((Counter(p) & Counter(g)).values())
    if not common: return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)

@torch.no_grad()
def eval_gen_em(examples, max_new=GEN_EM_MAXNEW):
    """Greedy free-gen → (EM, token-F1, n). Numeric gold (GSM8K, many DROP): EM on final number, F1==EM.
    Span gold (DROP): normalized first-line EM + SQuAD token-F1 (F1 is the metric that matters for DROP)."""
    import re
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    model.eval(); correct = f1sum = 0.0; total = 0
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for prompt, answer in examples[:GEN_EM_N]:
        gold = answer.replace(EOS, "").strip()
        p_ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)
        if p_ids.shape[1] >= MAX_LEN: continue
        gen = model.generate(p_ids, max_new_tokens=max_new, do_sample=False, pad_token_id=pad)
        out = tok.decode(gen[0][p_ids.shape[1]:], skip_special_tokens=True)
        gnum = _last_num(gold)
        if gnum is not None:                                   # numeric gold → compare final numbers
            onum = _last_num(out)
            ok = onum is not None and abs(float(onum) - float(gnum)) < 1e-6
            f1 = float(ok)                                     # numeric: F1 == EM
        else:                                                  # span gold → normalized first-line match
            first = out.split("\n")[0]
            ok = norm(first) == norm(gold)
            f1 = _tok_f1(first, gold)                          # span: SQuAD token-F1
        correct += int(ok); f1sum += f1; total += 1
    return (correct / total, f1sum / total, total) if total else (None, None, 0)

def eval_acc_all():
    """{leg}_acc dict: BBH (gen exact-match) + each MCQ EXTRA leg with choices (cloze choice-scoring)."""
    d = {}
    a, _ = eval_gen_acc(BBH_EVAL)
    if a is not None: d["bbh_acc"] = a
    if any(BBH_META):                                    # MC cloze over BBH-DEV: choice-acc + gold content-NLL
        mc, _, cn = eval_mcq_acc(BBH_EVAL, BBH_META)       # *_mc_cloze_acc / *_cloze_nll — content-scored twins of
        if mc is not None: d["bbh_mc_cloze_acc"] = mc      # the letter-scored bbh_acc / bbh_nll (store both, choose later)
        if cn is not None: d["bbh_cloze_nll"] = cn
    if BBH_TEST_EVAL:
        sa, _ = eval_gen_acc(BBH_TEST_EVAL)
        if sa is not None: d["bbh_test_acc"] = sa
        if any(BBH_TEST_META):                           # MC cloze over BBH-TEST: choice-acc + gold content-NLL
            smc, _, scn = eval_mcq_acc(BBH_TEST_EVAL, BBH_TEST_META)
            if smc is not None: d["bbh_test_mc_cloze_acc"] = smc
            if scn is not None: d["bbh_test_cloze_nll"] = scn
    for _nm, _ex in EXTRA_EVALS.items():
        _m = EXTRA_META.get(_nm)
        if _m and any(_m):
            acc, _, _ = eval_mcq_acc(_ex, _m)             # (MMLU cloze-NLL already logged via eval_qa; ignore here)
            if acc is not None: d[f"{_nm}_acc"] = acc
        elif _nm in _GEN_EM_LEGS:                          # free-gen: greedy EM + token-F1 (nll logged via eval_qa)
            em, f1, _ = eval_gen_em(_ex)
            if em is not None: d[f"{_nm}_em"] = em
            if f1 is not None: d[f"{_nm}_f1"] = f1
    model.train()
    return d

def save_perex(key, perex):
    if not LOG_PER_EX: return
    data = json.loads(PEREX_FILE.read_text()) if PEREX_FILE.exists() else {}
    data[key] = perex
    PEREX_FILE.write_text(json.dumps(data))

def sat_eval_rows(task):
    """First SAT_NEVAL cached TaskRows for token accuracy + native score_answer reward.
    Held-out from training ONLY when SAT_HOLDOUT=1 (then _train_aux drops these rows); otherwise they
    are ALSO in the training cycle → the resulting reward/acc is an in-training-fit signal, not held-out."""
    if task == "__COLLECTION__":            # pooled collection arm → round-robin a sample across ALL tasks
        lists = [v for v in _taskrow_aux.values() if v]     # (else reward_final stays None for every collection run)
        pool, i, cap = [], 0, max(SAT_NEVAL, 200)           # pool wide so REWARD_MODE filter still leaves >=5
        while lists and len(pool) < cap:
            grew = False
            for v in lists:
                if i < len(v):
                    pool.append(v[i]); grew = True
                    if len(pool) >= cap: break
            if not grew: break
            i += 1
        return pool
    return list(_taskrow_aux.get(task, [])[:SAT_NEVAL])

@torch.no_grad()
def token_acc(rows):
    """Teacher-forced argmax==gold over answer tokens (TRL's mean_token_accuracy)."""
    model.eval(); correct = total = 0
    for r in rows:
        prompt, answer = r["prompt"], r["answer"]
        p_ids = tok(f"{prompt}\n", add_special_tokens=False).input_ids
        a_ids = tok(f"{answer}{EOS}", add_special_tokens=False).input_ids
        if len(p_ids) + len(a_ids) > MAX_LEN or not a_ids: continue
        ids = torch.tensor([p_ids + a_ids], device=DEVICE)
        start = len(p_ids) - 1
        pred = model(ids).logits[0][start:-1].argmax(-1)
        gold = ids[0, start + 1:]
        correct += (pred == gold).sum().item(); total += gold.numel()
    model.train(); return correct / max(total, 1)

# Native task scorer (reasoning_core.score_answer / rg fallback), reused from zero_shot_eval so we don't
# duplicate the rc/rg routing. Import both ways: script run puts task_diagnostics/ on sys.path (bare
# name), package run needs the qualified name. Cheap: zero_shot_eval's top imports are stdlib only.
try:
    from zero_shot_eval import score_native as _score_native
except Exception:
    try:
        from task_diagnostics.zero_shot_eval import score_native as _score_native
    except Exception:
        _score_native = None

def _reward_one(row, gen):
    """Per-example native reward, or None when the scorer cannot evaluate this row."""
    return None if _score_native is None else _score_native(row, gen)

@torch.no_grad()
def free_gen_reward(rows):
    """Greedy free-generation reward over held-out rows via native score_answer. Same prompt format as
    training (`{prompt}\\n` → answer). Logged by default at BEGIN + END of every task's training."""
    if not rows: return None
    if REWARD_MODE not in ("", "all"):     # score_answer is only meaningful on the generate-the-answer mode
        instruct = [r for r in rows if (r.get("mode") or "") == REWARD_MODE]
        if len(instruct) >= 5: rows = instruct     # else keep all (local/cache rows carry no mode field)
    eos_id = tok.eos_token_id
    model.eval(); tot = 0.0; n = 0
    for r in rows[:REWARD_NEVAL]:
        prompt, answer = r["prompt"], r["answer"]
        p_ids = tok(f"{prompt}\n", add_special_tokens=False).input_ids
        a_ids = tok(f"{answer}{EOS}", add_special_tokens=False).input_ids
        if not a_ids or len(p_ids) >= MAX_LEN: continue
        cap = min(len(a_ids) + 8, REWARD_MAXTOK, MAX_LEN - len(p_ids))
        ids = torch.tensor([p_ids], device=DEVICE)
        out = model.generate(ids, max_new_tokens=cap, do_sample=False,
                             pad_token_id=eos_id, eos_token_id=eos_id)
        gen = tok.decode(out[0][len(p_ids):], skip_special_tokens=True)
        score = _reward_one(r, gen)
        if score is not None:
            tot += float(score); n += 1
    model.train(); return (tot / n) if n else None

class SatCurveCB(TrainerCallback):
    def __init__(self, rows, curve): self.rows, self.curve = rows, curve
    def on_step_end(self, args, state, control, **kw):
        if state.global_step % SAT_EVERY == 0:
            self.curve.append([state.global_step, token_acc(self.rows)])

class CkptEvalCB(TrainerCallback):
    """Eval BBH/Dolci/FW every CKPT_EVAL steps during training → budget-convergence curve."""
    def __init__(self, curve): self.curve = curve
    def on_step_end(self, args, state, control, **kw):
        if CKPT_EVAL and state.global_step % CKPT_EVAL == 0:
            (b, d, f, _fl), _ = eval_all()
            self.curve.append([state.global_step, b, d, f])

def save_ckpt(key, curve):
    data = json.loads(CKPT_FILE.read_text()) if CKPT_FILE.exists() else \
        {"main": MAIN_DATA, "init": _init_tag, "seed": SEED, "ckpt_every": CKPT_EVAL, "curves": {}}
    data["curves"][key] = curve
    CKPT_FILE.write_text(json.dumps(data))

def derive_sat(curve):
    if not curve: return {}
    accs = [a for _, a in curve]; final = accs[-1]; thr = 0.95 * final
    sat = next((s for s, a in curve if a >= thr), curve[-1][0])
    return {"acc0": curve[0][1], "acc_final": final,
            "auc": sum(accs) / len(accs), "sat_step": sat, "n_points": len(curve)}

def save_sat(task, rec):
    data = json.loads(SAT_FILE.read_text()) if SAT_FILE.exists() else \
        {"aux": AUX_DATASET, "main": MAIN_DATA, "init": _init_tag, "seed": SEED,
         "train_steps": TRAIN_STEPS, "mix_aux": MIX_AUX, "sat_every": SAT_EVERY,
         "oversample": OVERSAMPLE, "token_ratio": TOKEN_RATIO, "tr_stats": TR_STATS, "tasks": {}}
    data["tasks"][task] = rec
    SAT_FILE.write_text(json.dumps(data, indent=2))


# ── Training-data builders (parameterized by MAIN_DATA) ─────────────────────────
def _fmt_qa(prompt, answer):
    """prompt/completion (answer-only loss) when COMPLETION_ONLY, else packed text.
    Format matches run_sft.py get_formatter: prompt = f"{prompt}\n", completion = answer+EOS."""
    if COMPLETION_ONLY:
        return {"prompt": f"{prompt}\n", "completion": f"{answer}{EOS}"}
    return {"text": f"{prompt}\n{answer}{EOS}"}

def _fmt_text(text):
    if COMPLETION_ONLY:                 # raw text: matches run_sft fw (prompt="")
        return {"prompt": "", "completion": text + EOS}
    return {"text": text + EOS}

def main_gen():
    if MAIN_LOCAL:                               # stream-free: cycle the local cache (no HF stream)
        import itertools as _it
        rows = _read_jsonl(Path(MAIN_LOCAL) / f"{MAIN_DATA}_main.jsonl")
        for r in _it.cycle(rows):
            # Dispatch by ROW content, not the global MAIN_DATA, so a BLENDED main jsonl that mixes
            # qa rows ({prompt,answer}) and raw-text rows ({text}) trains each with its correct loss
            # mask (answer-only vs full-text). Single-main jsonls are unaffected (homogeneous rows).
            if r.get("answer"):
                yield _fmt_qa(r.get("prompt", ""), r["answer"])
            elif r.get("text"):
                yield _fmt_text(r["text"][:1200])
        return
    if MAIN_DATA == "dolci":
        for x in load_dataset("tasksource/dolci-instruct", split="train", streaming=True):
            if not x.get("answer"): continue
            yield _fmt_qa(x["prompt"], x["answer"])
    elif MAIN_DATA == "flan":
        for x in load_dataset("tasksource/flan", split="train", streaming=True):
            if not x.get("answer"): continue
            yield _fmt_qa(x["prompt"], x["answer"])
    else:  # fw
        for x in load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True):
            yield _fmt_text(x["text"][:1200])

TASKROW_CACHE = os.environ.get("TASKROW_CACHE", "")  # canonical TaskRow Parquet cache
if not TASKROW_CACHE:
    raise SystemExit("TASKROW_CACHE is required; build one with `python -m task_diagnostics.cache build`.")

def _drop_overlong(by_task):
    """Discard aux rows whose formatted (prompt+answer) exceeds MAX_LEN under the MODEL tokenizer.
    Under completion-only loss, packing (bfd) TRUNCATES an over-length row keep_start → the answer tail
    is dropped and the row trains on nothing meaningful. Discarding is correct; clipping is not.
    Default ON (DROP_OVERLONG=0 to keep legacy clip behavior). LEN_MARGIN reserves special-token headroom."""
    if os.environ.get("DROP_OVERLONG", "1") == "0":
        return by_task
    limit = MAX_LEN - int(os.environ.get("LEN_MARGIN", "8"))
    kept, dropped, gone = {}, 0, []
    for t, rows in by_task.items():
        pl = tok([f"{r.get('prompt','')}\n" for r in rows], add_special_tokens=True)["input_ids"]
        al = tok([f"{r.get('answer','')}" for r in rows], add_special_tokens=False)["input_ids"]
        keep = [r for r, p, a in zip(rows, pl, al) if len(p) + len(a) + 1 <= limit]  # +1 ≈ EOS
        dropped += len(rows) - len(keep)
        if keep:
            kept[t] = keep
        else:
            gone.append(t)
    if dropped:
        msg = f"✂️  DROP_OVERLONG: dropped {dropped} aux rows > {limit} tok (of {sum(len(v) for v in by_task.values())})"
        if gone:
            msg += f"; {len(gone)} task(s) fully removed: {sorted(gone)}"
        print(msg)
    return kept


def _load_taskrow_aux(path):
    try:
        from task_diagnostics.cache import load_task_rows
    except Exception:
        from cache import load_task_rows
    wanted = {t for t in (ALL_TASKS + PEER_TASKS + GROUP_TASKS)
              if t not in ("__ALLMIX__", "__GROUP__", "__COLLECTION__")}
    # COLLECTION pools every task in the cache → load them all (wanted empties to None below).
    by_task = {}
    for row in load_task_rows(path=path, tasks=sorted(wanted) or None):
        d = row.to_dict()
        if MODE_FILTER and (d.get("mode") or "") != MODE_FILTER:
            continue
        if not _level_ok(d):
            continue
        by_task.setdefault(d["task"], []).append(d)
    return _drop_overlong(by_task)

_taskrow_aux = _load_taskrow_aux(TASKROW_CACHE)
print(f"📦 TaskRow cache: {sum(len(v) for v in _taskrow_aux.values())} rows "
      f"across {len(_taskrow_aux)} tasks from {TASKROW_CACHE}")

# SAT_HOLDOUT: reserve the first SAT_NEVAL rows/task for token_acc + reward eval ONLY, and EXCLUDE them
# from the training stream, so saturation/reward measure held-out GENERALIZATION rather than in-training
# learnability/memorization. Default OFF for backward-compat (prior runs trained on those rows). When OFF,
# the reward/saturation column is an IN-TRAINING fit signal — not held-out. Influence legs (BBH/MMLU/mbpp/
# FW/dolci) are external and unaffected either way.
SAT_HOLDOUT = os.environ.get("SAT_HOLDOUT", "0") != "0"
_train_aux = ({t: rows[SAT_NEVAL:] for t, rows in _taskrow_aux.items()} if SAT_HOLDOUT else _taskrow_aux)
if SAT_HOLDOUT:
    print(f"🚧 SAT_HOLDOUT: first {SAT_NEVAL} rows/task held out of training → reward/acc are held-out")

def _cycle_rows(rows):
    import itertools as _it
    for r in _it.cycle(rows):
        if r.get("answer"):
            yield _fmt_qa(r.get("prompt", ""), r["answer"])

def rc_gen_factory(task_name):
    return lambda: _cycle_rows(_train_aux.get(task_name, []))

def all_aux_gen():
    """Every aux example (all tasks), natural rate — the 'full mixture' component."""
    yield from _cycle_rows([r for rs in _train_aux.values() for r in rs])

def all_aux_gen_mode(mode):
    """All-tasks aux stream restricted to one answer mode (for MODE_MIX full-mixture blends)."""
    def g():
        yield from _cycle_rows([r for rs in _train_aux.values() for r in rs
                                if (r.get("mode") or "") == mode])
    return g

def peer_aux_gen():
    """PEER_MIX background: only the fixed N_PEERS sampled tasks."""
    peers = list(PEER_TASKS)
    yield from _cycle_rows([r for t in peers for r in _train_aux.get(t, [])])

def group_aux_gen():
    """GROUP mode: pool the CHOSEN GROUP_TASKS into ONE aux arm (round-robin, equal share each)."""
    grp = list(GROUP_TASKS)
    yield from _cycle_rows([r for t in grp for r in _train_aux.get(t, [])])

# ── TOKEN_RATIO: derive the aux example-prob that matches (prompt+answer) token dose ──────
# Exact under packing: a packed stream is just the concatenation of the sampled example stream, so
#   aux_token_frac = p·L_aux / (p·L_aux + (1-p)·L_main).  Set aux_token_frac = MIX_AUX and solve:
#   p = MIX_AUX / [ (L_aux/L_main)·(1-MIX_AUX) + MIX_AUX ].   p→MIX_AUX when L_aux==L_main.
def _mean_qa_tok_len(rows, cap=4000):
    if len(rows) > cap:
        import random as _r; rows = _r.Random(SEED).sample(rows, cap)
    pl = tok([f"{r.get('prompt','')}\n" for r in rows], add_special_tokens=True)["input_ids"]
    al = tok([f"{r.get('answer','')}"   for r in rows], add_special_tokens=False)["input_ids"]
    return sum(len(p) + len(a) + 1 for p, a in zip(pl, al)) / max(1, len(rows))  # +1 ≈ EOS

def _mean_main_tok_len(n=1500):
    tot = c = 0
    for ex in main_gen():
        ids = (tok(ex["prompt"], add_special_tokens=True)["input_ids"]
               + tok(ex["completion"], add_special_tokens=False)["input_ids"]) if COMPLETION_ONLY \
              else tok(ex["text"], add_special_tokens=True)["input_ids"]
        tot += len(ids); c += 1
        if c >= n: break
    return tot / max(1, c)

if TOKEN_RATIO:
    _L_aux  = _mean_qa_tok_len([r for rs in _train_aux.values() for r in rs])
    _L_main = _mean_main_tok_len()
    _ratio  = _L_aux / max(1e-9, _L_main)
    AUX_P   = MIX_AUX / (_ratio * (1.0 - MIX_AUX) + MIX_AUX)
    TR_STATS = {"token_ratio": True, "target_aux_token_frac": MIX_AUX,
                "L_aux": round(_L_aux, 2), "L_main": round(_L_main, 2),
                "len_ratio": round(_ratio, 4), "aux_example_prob": round(AUX_P, 5)}
    print(f"🎯 TOKEN_RATIO on: L_aux={_L_aux:.1f} tok  L_main={_L_main:.1f} tok  ratio={_ratio:.3f} "
          f"→ aux example-prob {AUX_P:.4f}  (target aux TOKEN frac {MIX_AUX:.2f}; "
          f"example-matched prob would be {MIX_AUX:.2f})", flush=True)
else:
    AUX_P, TR_STATS = MIX_AUX, {"token_ratio": False, "aux_example_prob": MIX_AUX}
MAIN_P = 1.0 - AUX_P

def baseline_ds():
    if PEER_MIX:     # realistic K-peer background: main + peers at MIX (peer set fixed)
        return interleave_datasets(
            [IterableDataset.from_generator(main_gen),
             IterableDataset.from_generator(peer_aux_gen)],
            probabilities=[1.0 - MIX_AUX, MIX_AUX],
            seed=SEED, stopping_strategy="first_exhausted",
        )
    if OVERSAMPLE:   # full mixture: main + all-aux at MIX
        return interleave_datasets(
            [IterableDataset.from_generator(main_gen),
             IterableDataset.from_generator(all_aux_gen)],
            probabilities=[1.0 - MIX_AUX, MIX_AUX],
            seed=SEED, stopping_strategy="first_exhausted",
        )
    return IterableDataset.from_generator(main_gen)

def mixed_ds(task_name):
    if task_name == "__COLLECTION__":   # whole collection pooled as ONE aux arm vs main-only baseline
        return interleave_datasets(     # probs = [MAIN_P, AUX_P]; token-matched iff TOKEN_RATIO, else example-matched
            [IterableDataset.from_generator(main_gen),
             IterableDataset.from_generator(all_aux_gen)],
            probabilities=[MAIN_P, AUX_P],
            seed=SEED, stopping_strategy="first_exhausted",
        )
    if task_name == "__ALLMIX__":   # full-mixture answer-format ablation: main + blended all-aux
        parts = [IterableDataset.from_generator(main_gen)]
        probs = [1.0 - MIX_AUX]
        for m, w in _mode_mix:
            parts.append(IterableDataset.from_generator(all_aux_gen_mode(m)))
            probs.append(MIX_AUX * w)
        return interleave_datasets(parts, probabilities=probs, seed=SEED,
                                   stopping_strategy="first_exhausted")
    # x_gen = the "X" arm: a single task, or (GROUP mode) the pooled chosen set — same downstream shape.
    x_gen = group_aux_gen if task_name == "__GROUP__" else rc_gen_factory(task_name)
    if task_name == "__GROUP__" and not OVERSAMPLE:   # group pooled, alone (no background) at total MIX
        return interleave_datasets(
            [IterableDataset.from_generator(main_gen),
             IterableDataset.from_generator(x_gen)],
            probabilities=[1.0 - MIX_AUX, MIX_AUX],
            seed=SEED, stopping_strategy="first_exhausted",
        )
    if PEER_MIX:     # upweight X (task or group) on top of fixed K-peer background; total aux = MIX
        half = MIX_AUX / 2.0
        return interleave_datasets(
            [IterableDataset.from_generator(main_gen),
             IterableDataset.from_generator(x_gen),
             IterableDataset.from_generator(peer_aux_gen)],
            probabilities=[1.0 - MIX_AUX, half, half],
            seed=SEED, stopping_strategy="first_exhausted",
        )
    if OVERSAMPLE:   # upweight X (task or group) on top of the FULL all-rc background; total aux = MIX
        x_share  = MIX_AUX * OVS_X_FRAC          # extra dose of the measured task
        bg_share = MIX_AUX * (1.0 - OVS_X_FRAC)  # all-rc background (already contains X at ~1/N)
        return interleave_datasets(
            [IterableDataset.from_generator(main_gen),
             IterableDataset.from_generator(x_gen),
             IterableDataset.from_generator(all_aux_gen)],
            probabilities=[1.0 - MIX_AUX, x_share, bg_share],
            seed=SEED, stopping_strategy="first_exhausted",
        )
    return interleave_datasets(
        [IterableDataset.from_generator(main_gen),
         IterableDataset.from_generator(x_gen)],
        probabilities=[1.0 - MIX_AUX, MIX_AUX],
        seed=SEED, stopping_strategy="first_exhausted",
    )

def train_on(ds, callbacks=None):
    cfg = dict(
        output_dir="/tmp/influence_sft", overwrite_output_dir=True,
        max_steps=TRAIN_STEPS, per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR, weight_decay=0.01,
        max_length=MAX_LEN, packing=(os.environ.get("PACKING","1")!="0"),
        bf16=(DTYPE == torch.bfloat16), report_to="none",
        save_strategy="no", logging_steps=1000, dataset_num_proc=1,
        seed=SEED, disable_tqdm=True,
        optim=os.environ.get("OPTIM", "adamw_torch"),  # OPTIM=adamw_bnb_8bit to fit ~1.7B on a 23G GPU
    )
    if os.environ.get("GRAD_CKPT", "0") == "1":
        cfg["gradient_checkpointing"] = True
        cfg["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if COMPLETION_ONLY:                 # mask prompt — matches run_sft production
        cfg["completion_only_loss"] = True
    else:
        cfg["dataset_text_field"] = "text"
    import dataclasses   # trl versions differ in SFTConfig kwargs (e.g. newer trl dropped overwrite_output_dir)
    _valid = {f.name for f in dataclasses.fields(SFTConfig)}
    _drop = [k for k in list(cfg) if k not in _valid]
    for k in _drop: cfg.pop(k)
    if _drop: print(f"  (SFTConfig: dropped kwargs unsupported by this trl: {_drop})", flush=True)
    SFTTrainer(model=model, processing_class=tok, train_dataset=ds,
               args=SFTConfig(**cfg), callbacks=callbacks).train()


# ── Emergency-rerun checkpoints (opt-in; heavy runs only) ────────────────────────
# Save the FINAL trained model per arm so a new eval leg can be re-scored WITHOUT retraining.
# OFF unless SAVE_CKPT_DIR is set. Weights-only (no optimizer). Hard disk guard: never write if the
# target FS has < SAVE_CKPT_MIN_FREE_GB free (protects the shared Lille NFS, ~95% full).
SAVE_CKPT_DIR       = os.environ.get("SAVE_CKPT_DIR", "").strip()
SAVE_CKPT_MIN_FREE_GB = float(os.environ.get("SAVE_CKPT_MIN_FREE_GB", "300"))
def _save_ckpt_model(tag):
    if not SAVE_CKPT_DIR:
        return
    import shutil
    slug = MODEL_NAME.replace("/", "-")
    dest = Path(SAVE_CKPT_DIR) / f"{slug}_{MAIN_DATA}_S{SEED}_T{TRAIN_STEPS}" / str(tag)
    try:
        free_gb = shutil.disk_usage(SAVE_CKPT_DIR).free / 2**30
        if free_gb < SAVE_CKPT_MIN_FREE_GB:
            print(f"⚠️  SKIP ckpt {tag}: {free_gb:.0f}GB free < {SAVE_CKPT_MIN_FREE_GB}GB guard", flush=True)
            return
        dest.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(dest, safe_serialization=True)   # weights only
        tok.save_pretrained(dest)
        print(f"💾 ckpt saved: {dest}  ({free_gb:.0f}GB free)", flush=True)
    except Exception as e:
        print(f"⚠️  ckpt save failed for {tag}: {e}", flush=True)


# ── Resume support ──────────────────────────────────────────────────────────────
if OUT_FILE.exists():
    results = json.loads(OUT_FILE.read_text())
    print(f"♻️  Resuming — {len(results.get('tasks', {}))} task(s) done\n")
else:
    results = {"seed": SEED, "train_steps": TRAIN_STEPS, "mix_aux": MIX_AUX,
               "main_data": MAIN_DATA, "from_scratch": FROM_SCRATCH,
               "lr": LR, "batch": BATCH, "model": MODEL_NAME,
               "max_len": MAX_LEN, "packing": os.environ.get("PACKING", "1") != "0",
               "drop_overlong": os.environ.get("DROP_OVERLONG", "1") != "0",
               "token_ratio": TOKEN_RATIO, "tr_stats": TR_STATS,
               "oversample": OVERSAMPLE, "peer_mix": PEER_MIX,
               "peer_tasks": PEER_TASKS, "n_peers": N_PEERS if PEER_MIX else 0,
               "tasks": {}, "baseline": None, "pretrained_only": None}

# Shared baseline cache: the pretrained-only eval and the main-only baseline depend ONLY on
# (model, MAIN_DATA, SEED, TRAIN_STEPS) — never on the aux collection/task. Point BASELINE_CACHE at a
# path keyed by those, and every collection/task run at the same setting reuses it instead of retraining
# the (expensive at high step counts) baseline. Opt-in: unset → unchanged behavior.
BASELINE_CACHE = os.environ.get("BASELINE_CACHE", "")
if BASELINE_CACHE and results.get("baseline") is None and Path(BASELINE_CACHE).exists():
    try:
        _bc = json.loads(Path(BASELINE_CACHE).read_text())
        results["baseline"] = _bc["baseline"]
        if results.get("pretrained_only") is None:
            results["pretrained_only"] = _bc.get("pretrained_only")
        print(f"♻️  Baseline + pretrained-only loaded from shared cache {BASELINE_CACHE}")
    except Exception as e:
        print(f"⚠️  BASELINE_CACHE read failed ({e}); recomputing")

# Pretrained-only sanity
if results.get("pretrained_only") is None:
    (pb, pd, pf, pex), px = eval_all(); save_perex("pretrained_only", px)
    print(f"📏 Pre-training eval: BBH={pb:.4f}  Dolci={pd:.4f}  FW={pf:.4f}"
          + "".join(f"  {k}={v:.4f}" for k, v in pex.items()) + "\n")
    results["pretrained_only"] = {"bbh_nll": pb, "dolci_nll": pd, "fw_nll": pf}
    for k, v in pex.items(): results["pretrained_only"][f"{k}_nll"] = v
    OUT_FILE.write_text(json.dumps(results, indent=2))

# Baseline: train on MAIN_DATA only
if results.get("baseline") is None:
    print(f"🏃 Baseline: train on {MAIN_DATA} only ({TRAIN_STEPS} steps)…")
    t0 = time.time(); reset_model()
    _bl_cb = None
    if CKPT_EVAL:
        _bl_curve = []; _bl_cb = [CkptEvalCB(_bl_curve)]
    train_on(baseline_ds(), callbacks=_bl_cb)
    if CKPT_EVAL: save_ckpt("baseline", _bl_curve)
    _save_ckpt_model("baseline")
    (b_bbh, b_dolci, b_fw, b_ex), px = eval_all(); save_perex("baseline", px)
    dt = time.time() - t0
    print(f"  BBH={b_bbh:.4f}  Dolci={b_dolci:.4f}  FW={b_fw:.4f}"
          + "".join(f"  {k}={v:.4f}" for k, v in b_ex.items()) + f"  ({dt:.0f}s)\n")
    results["baseline"] = {"bbh_nll": b_bbh, "dolci_nll": b_dolci, "fw_nll": b_fw}
    for k, v in b_ex.items(): results["baseline"][f"{k}_nll"] = v
    if EVAL_ACC:
        try:
            for _k, _v in eval_acc_all().items(): results["baseline"][_k] = _v
            print("  acc: " + "  ".join(f"{k}={results['baseline'][k]:.3f}" for k in results['baseline'] if k.endswith('_acc')))
        except Exception as _ae:
            print(f"  ⚠ baseline accuracy eval failed ({type(_ae).__name__}: {_ae}) — NLL-only", flush=True)
    OUT_FILE.write_text(json.dumps(results, indent=2))
    if BASELINE_CACHE:  # persist for peer collection/task runs at this (model, main, seed, steps)
        Path(BASELINE_CACHE).write_text(json.dumps(
            {"baseline": results["baseline"], "pretrained_only": results["pretrained_only"]}, indent=2))
        print(f"💾 Baseline saved to shared cache {BASELINE_CACHE}")
else:
    b_bbh, b_dolci, b_fw = (results["baseline"]["bbh_nll"],
                            results["baseline"]["dolci_nll"],
                            results["baseline"]["fw_nll"])
    _extra_legs = (list(EXTRA_EVALS) + (["bbh_test"] if EVAL_BBH_TEST else [])
                   + (["bbh_open"] if EVAL_BBH_OPEN else [])
                   + (["bbh_test_open"] if (EVAL_BBH_OPEN and EVAL_BBH_TEST) else []))  # not in EXTRA_EVALS
    b_ex = {k: results["baseline"][f"{k}_nll"] for k in _extra_legs if f"{k}_nll" in results["baseline"]}
    print(f"📏 Baseline (cached): BBH={b_bbh:.4f}  Dolci={b_dolci:.4f}  FW={b_fw:.4f}"
          + "".join(f"  {k}={v:.4f}" for k, v in b_ex.items()) + "\n")


# Per-task loop
for i, task in enumerate(ALL_TASKS):
    if task in results["tasks"]:
        print(f"[{i+1}/{len(ALL_TASKS)}] {task} — already done"); continue
    if _taskrow_aux is not None and task not in ("__ALLMIX__", "__GROUP__", "__COLLECTION__") and not _taskrow_aux.get(task):
        print(f"[{i+1}/{len(ALL_TASKS)}] {task} — SKIP (no TaskRow cache rows)")
        continue
    t0 = time.time(); reset_model()
    try:  # one task's transient failure (e.g. HF streaming timeout) must not kill the whole 51-task run
        sat_cbs, sat_curve, reward0 = None, None, None
        sat_rows = sat_eval_rows(task) if (LOG_SAT or LOG_REWARD) else []
        if LOG_SAT and len(sat_rows) >= 5:
            sat_curve = [[0, token_acc(sat_rows)]]
            sat_cbs = [SatCurveCB(sat_rows, sat_curve)]
        if LOG_REWARD and len(sat_rows) >= 5:
            reward0 = free_gen_reward(sat_rows)   # BEGIN free-gen reward (before training)
        ckpt_curve = None
        if CKPT_EVAL:
            ckpt_curve = []; sat_cbs = (sat_cbs or []) + [CkptEvalCB(ckpt_curve)]
        train_on(mixed_ds(task), callbacks=sat_cbs)
        if ckpt_curve is not None: save_ckpt(task, ckpt_curve)
        _save_ckpt_model(COLLECTION or task)
        (m_bbh, m_dolci, m_fw, m_ex), px = eval_all(); save_perex(task, px)
        reward_final = free_gen_reward(sat_rows) if (reward0 is not None) else None  # END free-gen reward
        if sat_curve is not None or reward0 is not None:
            rec = {}
            if sat_curve is not None:
                rec["curve"] = sat_curve; rec.update(derive_sat(sat_curve))
            if reward0 is not None:
                rec["reward0"] = reward0; rec["reward_final"] = reward_final
                rec["reward_gain"] = (reward_final - reward0) if reward_final is not None else None
            save_sat(task, rec)
        dt = time.time() - t0
        results["tasks"][task] = {
            "bbh_nll": m_bbh, "dolci_nll": m_dolci, "fw_nll": m_fw,
            "bbh_delta":   m_bbh - b_bbh,
            "dolci_delta": m_dolci - b_dolci,
            "fw_delta":    m_fw - b_fw,
            "time_s":      round(dt, 1),   # wall-clock per task (train + all-leg eval)
        }
        if reward0 is not None:   # native score_answer solve-rate, begin -> end (cheap + informative)
            results["tasks"][task]["reward0"] = reward0
            results["tasks"][task]["reward_final"] = reward_final
        for k, v in m_ex.items():
            if k in b_ex:
                results["tasks"][task][f"{k}_nll"]   = v
                results["tasks"][task][f"{k}_delta"] = v - b_ex[k]
        if EVAL_ACC:   # held-out accuracy + Δaccuracy (arm − baseline) for BBH + MCQ legs
            for _k, _v in eval_acc_all().items():
                results["tasks"][task][_k] = _v
                _bk = results["baseline"].get(_k)
                if _bk is not None: results["tasks"][task][f"{_k}_delta"] = _v - _bk
        OUT_FILE.write_text(json.dumps(results, indent=2))
        print(f"[{i+1:2d}/{len(ALL_TASKS)}] {task:<26s}  "
              f"BBH Δ={m_bbh-b_bbh:+.4f}  Dolci Δ={m_dolci-b_dolci:+.4f}  "
              f"FW Δ={m_fw-b_fw:+.4f}"
              + "".join(f"  {k}Δ={m_ex[k]-b_ex[k]:+.4f}" for k in m_ex if k in b_ex)
              + (f"  solve {reward0:.2f}->{reward_final:.2f}" if reward0 is not None and reward_final is not None else "")
              + f"  ({dt:.0f}s)")
    except Exception as _e:
        import traceback
        print(f"[{i+1:2d}/{len(ALL_TASKS)}] {task:<26s}  ERROR ({type(_e).__name__}: {_e}) "
              f"— skipping, will be recomputed on resume", flush=True)
        traceback.print_exc()
        continue

print(f"\n✅ Done → {OUT_FILE}")
