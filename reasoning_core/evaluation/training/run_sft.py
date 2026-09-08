"""
for budget in 300_000_000 1_000_000_000; do   for r in 0.1 0.0 0.25 0.4 0.05; do     python run_sft.py --aux_ratio "$r" --token_budget "$budget";   done; done
"""
# run_sft.py
import os, tempfile
from pathlib import Path

TMP_ROOT = Path(os.environ.get("RC_TMP", str(Path.home() / "tmp"))).expanduser()
SAFE_TMP = os.environ.get("SAFE_TMP", str(TMP_ROOT / "cache"))
HF_CACHE = os.environ.get("HF_CACHE", str(TMP_ROOT / "hf_cache"))
os.environ['HF_HOME'] = HF_CACHE
os.environ['HF_DATASETS_CACHE'] = HF_CACHE
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ['WANDB__SERVICE_WAIT']="120"
os.environ['WANDB_INIT_TIMEOUT']="200"
for k in ("TMPDIR", "TEMP", "TMP"):
    os.environ[k] = SAFE_TMP
os.makedirs(SAFE_TMP, exist_ok=True)
os.makedirs(HF_CACHE, exist_ok=True)
tempfile.tempdir = SAFE_TMP

from itertools import islice
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import logging, argparse, torch, wandb, ast, json, hashlib, shutil, copy
from faker import Faker
from datasets import (
    load_dataset, concatenate_datasets, interleave_datasets,
    Dataset, IterableDataset, disable_caching,
)
from huggingface_hub import dataset_info
import hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, AutoConfig
from trl import SFTConfig
from tabulate import tabulate
from reasoning_core.training.controlled_experiment import add_control_args, row_filter, wrap_aux_dataset_for_control
from reasoning_core.training.intrinsic_rewards import load_intrinsic_eval_split, log_intrinsic_task_rewards
from reasoning_core.training.local_metrics import LocalMetricsCallback, LocalMetricsSink
from reasoning_core.training.optimizers import add_optimizer_args, create_optimizer_and_scheduler, trainer_cls_for_optimizer
from reasoning_core.training.source_signals import SourceDataCollator, add_source, pack_by_source
import socket, threading, time, atexit

disable_caching()
logging.getLogger("trl.trainer.sft_trainer").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


# --- ⚙️ Config ---
parse_num = lambda s: int(float(s[:-1]) * {'K': 1e3, 'M': 1e6, 'B': 1e9}[s[-1].upper()]) if s[-1].isalpha() else int(s)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="smol135")
parser.add_argument('--token_budget', type=parse_num, default=100_000_000)
parser.add_argument('--aux_ratio', type=float, default=0.2)
parser.add_argument('--phase_2_ratio', type=float, default=0.0)
parser.add_argument('--main_data', type=str, default="dolci", choices=["fw", "synth", "dolci", "flan"])
parser.add_argument('--aux_data', type=str, default="rc")
parser.add_argument('--max_length', type=int, default=1024)
parser.add_argument('--decay', type=float, default=0.01)
parser.add_argument('--from_scratch', type=ast.literal_eval, default=True)
parser.add_argument('--aux_version', type=str, default="rc11")
parser.add_argument('--script_version', type=str, default="25")
parser.add_argument('--aux_token', type=str, default="")
parser.add_argument('--iterable_mode', type=ast.literal_eval, default=True)
parser.add_argument('--title', type=str, default='')
parser.add_argument("--experiment_name", type=str, default="")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_eval', type=int, default=10)
parser.add_argument('--target_effective_bs', type=int, default=64)
parser.add_argument('--per_device_train_batch_size', type=int, default=0)
parser.add_argument('--gradient_checkpointing', type=ast.literal_eval, default=False)
parser.add_argument("--save_steps", type=int, default=400)
parser.add_argument("--save_total_limit", type=int, default=1)
add_control_args(parser)
parser.add_argument("--eval_main_override", type=str, default="")
parser.add_argument("--eval_main_budget", type=int, default=50_000)
parser.add_argument("--eval_main_skip", type=int, default=1_000_000)
parser.add_argument("--eval_aux_budget", type=int, default=1_500_000)
parser.add_argument("--eval_aux_skip", type=int, default=100_000)
parser.add_argument("--eval_aux_max_scanned", type=int, default=50_000)
add_optimizer_args(parser)

DATA_MAP = {
    "fw": "HuggingFaceFW/fineweb-edu",
    "synth": "tasksource/SYNTH",
    "dolci": "tasksource/dolci-instruct",
    "rc": "reasoning-core/procedural-pile",
    "rg": "reasoning-core/reasoning-gym",
    "bp": "reasoning-core/procedural-warmup",
    "sl": "reasoning-core/synlogic",
    "flan": "tasksource/flan"
}

MODEL_MAP = {
    "monad": "PleIAs/Monad",
    "smol135": "HuggingFaceTB/SmolLM2-135M",
    "baguette": "PleIAs/Baguettotron",
    "h1": "tiiuae/Falcon-H1-Tiny-90M-Instruct",
    "ettin32": "jhu-clsp/ettin-decoder-32m",
    "microlm-5m": "sileod/microlm-ettin-swa-5m",
    "ettin68": "jhu-clsp/ettin-decoder-68m",
    "ettin150": "jhu-clsp/ettin-decoder-150m",
    "lfm350":"LiquidAI/LFM2.5-350M-Base"
}


def in_notebook():
    from IPython import get_ipython
    return getattr(get_ipython(), '__class__', None).__name__ == 'ZMQInteractiveShell'

args, _ = parser.parse_known_args()
args.n_eval = max(1, args.n_eval)
if args.main_data == "dolci":
    args.from_scratch = False
args.model_name = MODEL_MAP.get(args.model_name, args.model_name)
SPECIAL = args.aux_token

# Stage 2 requires knowing how many rows Stage 1 consumed — not possible upfront with iterable
if args.iterable_mode and args.phase_2_ratio > 0:
    print("⚠️  phase_2_ratio > 0 ignored in ITERABLE_MODE")
    args.phase_2_ratio = 0.0

info = dataset_info(DATA_MAP.get(args.aux_data, args.aux_data), files_metadata=True)
blobs = [f.blob_id for f in info.siblings if f.rfilename.endswith(('.parquet', '.jsonl', '.arrow')) and f.blob_id]
args.aux_hash = hashlib.md5("".join(blobs).encode()).hexdigest()[:8] if blobs else info.sha[:8]

print(tabulate(vars(args).items()))
print(f"📡 ITERABLE_MODE = {args.iterable_mode}")


# --- 💾 Checkpointing ---
def _args_hash(a):
    d = {k: str(v) for k, v in sorted(vars(a).items()) if k != "n_eval"}
    return hashlib.sha256(json.dumps(d).encode()).hexdigest()[:16]

def _run_name(seed_str):
    fake = Faker(); fake.seed_instance(int(hashlib.md5(seed_str.encode()).hexdigest(), 16))
    return f"{fake.color_name().lower()}-{fake.word()}-{int(hashlib.md5(seed_str.encode()).hexdigest()[:4], 16) % 1000}"

def save_ckpt(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))
    shutil.copy2(path, path.with_suffix(".bak.json"))

def load_ckpt(path):
    for p in [path, path.with_suffix(".bak.json")]:
        try: return json.loads(p.read_text())
        except (FileNotFoundError, json.JSONDecodeError): continue
    return None

run_hash = _args_hash(args)
b=f"{args.token_budget/1e6:.2f}M"
run_name = f"{_run_name(run_hash)}-b{b}-r{args.aux_ratio}"
ckpt_dir = Path("checkpoints") / run_hash
ckpt_file = ckpt_dir / "run_state.json"
ckpt = load_ckpt(ckpt_file)

# --- 🔒 NFS-safe cross-machine lock ---
import socket, threading, time, atexit
EXIT_CODE=99
def acquire_lock(lock_file: Path, stale_after: int = 300, heartbeat_every: int = 60):
    """Cross-machine lock via O_EXCL create + heartbeat. Exits if another runner is active."""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    stamp = lambda: f"{socket.gethostname()}:{os.getpid()}:{time.time()}"

    def lock_age():
        try: return time.time() - float(lock_file.read_text().rsplit(":", 1)[-1])
        except (FileNotFoundError, ValueError): return float("inf")

    def try_create():
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            lock_file.write_text(stamp())
            return True
        except FileExistsError:
            return False

    if not try_create():
        if lock_age() < stale_after:
            print(f"🔒 Active elsewhere: {lock_file.read_text()}. Exiting."); exit(EXIT_CODE)
        lock_file.unlink(missing_ok=True)                 # stale → clear and retry
        if not try_create():
            print(f"🔒 Lost race: {lock_file.read_text()}. Exiting."); exit(EXIT_CODE)

    atexit.register(lambda: lock_file.unlink(missing_ok=True))

    def heartbeat():
        while True:
            time.sleep(heartbeat_every)
            try: lock_file.write_text(stamp())
            except OSError: return
    threading.Thread(target=heartbeat, daemon=True).start()

#acquire_lock(ckpt_dir / "run.lock")


if ckpt and ckpt.get("hash") == run_hash:
    if ckpt.get("done"):
        print("✅ Run already completed with these args. Exiting.")
        exit(0)
    completed = set(ckpt.get("completed_stages", []))
    group_id = ckpt["group_id"]
    print(f"♻️  Resuming {run_name} — completed stages: {completed}")
else:
    completed = set()
    group_id = f"{args.main_data}+{args.aux_data}-{args.experiment_name}" if args.experiment_name else f"{args.main_data}+{args.aux_data}-{run_name}"
    save_ckpt({"hash": run_hash, "group_id": group_id,
               "args": {k: str(v) for k, v in vars(args).items()},
               "completed_stages": [], "done": False}, ckpt_file)


# --- 🧠 Model & Tokenizer ---
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
if not args.from_scratch:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype, device_map="auto", attn_implementation="sdpa")
else:
    cfg = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_config(cfg, dtype=dtype, attn_implementation="sdpa").to("cuda")

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
if tokenizer.bos_token is None:
    tokenizer.add_special_tokens({'bos_token': '<s>'})
    tokenizer.bos_token = '<s>'
    model.resize_token_embeddings(len(tokenizer))

if SPECIAL.strip():
    tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL]})
    model.resize_token_embeddings(len(tokenizer))

if getattr(tokenizer, "chat_template", None) is None:
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


# --- 📚 Data Loading ---
def get_formatter(data_key):
    if data_key == "fw":
        return lambda x: {"prompt": "", "completion": x["text"] + tokenizer.eos_token}

    def fmt(x):
        prefix = SPECIAL + "\n" if SPECIAL and data_key == "rc" else ""
        answer = x.get("answer")
        ex = {
            "prompt": prefix + f"Q: {x['prompt']}\nA:",
            "completion": " "+answer + tokenizer.eos_token,
        }
        ex.update({k: x[k] for k in ("task", "level", "mode") if x.get(k) not in (None, "")})
        source_task = _source_task(x)
        if source_task:
            ex["task"] = source_task
        return ex
    return fmt


def _source_task(row):
    raw = row.get("metadata", {}) if isinstance(row, dict) else {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    if not isinstance(raw, dict):
        return None
    return raw.get("source_dataset") or raw.get("task_name") or raw.get("rg_task")

# --- Materialized loader (original behavior; also used for evals) ---
def load_exact_tokens_materialized(key, budget, stream_skip=0, max_len=None, batch_size=1000):
    if budget <= 0:
        return None, 0

    hf_path = DATA_MAP.get(key, key)
    fmt_fn = get_formatter(key)
    print(f"⏳ [materialize] {key} ({hf_path}) [skip={stream_skip}] target={budget/1e6:.2f}M tokens...")
    stream = load_dataset(hf_path, split="train", streaming=True).skip(stream_skip)

    rows_consumed = 0
    tokens_total = 0

    def gen():
        nonlocal rows_consumed, tokens_total
        it = iter(stream)
        while tokens_total < budget:
            chunk = list(islice(it, batch_size))
            if not chunk: break
            rows_consumed += len(chunk)
            exs = [fmt_fn(r) for r in chunk]
            texts = [e['prompt'] + e['completion'] for e in exs]
            lens = tokenizer(texts, add_special_tokens=True, return_length=True,
                             return_attention_mask=False, return_token_type_ids=False)['length']
            for ex, L in zip(exs, lens):
                if max_len and L > max_len: continue
                tokens_total += L
                yield ex
                if tokens_total >= budget: return

    ds = Dataset.from_generator(gen)
    print(f"✅ Loaded {ds.num_rows} rows ({tokens_total} tokens).")
    return ds, rows_consumed


# --- Iterable loader: lazy, approximate tokens-via-chars ---
def load_exact_tokens_iterable(key, budget, max_len=None, chars_per_token=4.0, cycle=False):
    if budget <= 0:
        return None

    hf_path = DATA_MAP.get(key, key)
    fmt_fn = get_formatter(key)
    print(f"⏳ [iterable] {key} ({hf_path}) infinite stream (budget handled by max_steps, cycle={cycle})")

    max_chars = max_len * chars_per_token if max_len else None

    def gen():
        while True:
            stream = load_dataset(hf_path, split="train", streaming=True)
            for row in stream:
                ex = fmt_fn(row)
                if max_chars and (len(ex["prompt"]) + len(ex["completion"])) > max_chars:
                    continue
                yield ex
            if not cycle:
                break

    ds = IterableDataset.from_generator(gen)
    return ds


# --- 🔄 Training ---
_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
_model_params = sum(p.numel() for p in model.parameters()) / 1e6

available_memory_factor = max(1, min(16, round(
    2 * (_model_params / 68) * (24 / _total_gb)
)))

observed_memory_peak = 0.90
target_memory_peak = 0.90

available_memory_factor = max(1, min(16, round(
    available_memory_factor * observed_memory_peak / target_memory_peak
)))

target_effective_bs = max(1, args.target_effective_bs)
auto_per_device_bs = 1 << ((16 // available_memory_factor).bit_length() - 1)  # largest pow2 ≤ 16//factor
per_device_bs = args.per_device_train_batch_size or auto_per_device_bs
per_device_bs = max(1, min(per_device_bs, target_effective_bs))
grad_accum = max(1, target_effective_bs // per_device_bs)
eff_batch = per_device_bs * grad_accum
local_metrics_sink = LocalMetricsSink(
    ckpt_dir / "metrics.jsonl",
    run_hash=run_hash,
    group_id=group_id,
    script_args=args,
    eff_batch=eff_batch,
)


# --- 🧪 Experiment Setup ---

def clean_metric_key(x):
    return str(x).lower().replace("/", "_").replace(" ", "_").replace(".", "_")[:60]

def load_eval_split(key, budget=500_000, skip=100_000, group_by=("task", "level"), max_groups=200, max_examples_per_group=32, min_examples_per_group=8, max_scanned=50_000):
    fmt = get_formatter(key)
    stream = load_dataset(DATA_MAP.get(key, key), split="train", streaming=True).skip(skip)
    buckets, tokens = defaultdict(list), 0
    for i, row in enumerate(stream):
        if tokens >= budget or i >= max_scanned: break
        if len(buckets) >= max_groups and all(len(v) >= max_examples_per_group for v in buckets.values()): break
        vals = [row.get(k) for k in group_by]
        if any(v in (None, "") for v in vals): continue
        g = ".".join(clean_metric_key(v) for v in vals)
        if g not in buckets and len(buckets) >= max_groups: continue
        if len(buckets[g]) >= max_examples_per_group: continue
        ex = fmt(row)
        L = len(tokenizer(ex["prompt"] + ex["completion"]).input_ids)
        if L > args.max_length: continue
        buckets[g].append(ex); tokens += L
    
    result = {k: Dataset.from_list(v) for k, v in buckets.items() if len(v) >= min_examples_per_group}
    print(f"📊 eval buckets ({len(result)}/{len(buckets)} kept, {tokens} tokens): {sorted((k, len(v)) for k, v in result.items())}")
    return result

# Evals: always materialized (tiny, trainer needs __len__)
eval_key = args.eval_main_override or args.main_data
eval_main, _ = load_exact_tokens_materialized(
    eval_key, args.eval_main_budget, stream_skip=args.eval_main_skip, max_len=args.max_length)
eval_splits = load_eval_split(
    args.aux_data, budget=args.eval_aux_budget, skip=args.eval_aux_skip,
    max_groups=200, max_examples_per_group=32, min_examples_per_group=24,
    max_scanned=args.eval_aux_max_scanned)
intrinsic_eval_splits = load_intrinsic_eval_split(
    DATA_MAP.get(args.aux_data, args.aux_data), skip=args.eval_aux_skip, max_groups=200,
    max_examples_per_group=8, max_scanned=args.eval_aux_max_scanned)
main_evals = {"main": eval_main}
aux_tl_evals = {f"aux_tl/{k}": v for k, v in eval_splits.items()}
trainer_evals = main_evals if args.iterable_mode else {**main_evals, **aux_tl_evals}


aux_budget = int(args.token_budget * args.aux_ratio)

if args.iterable_mode:
    s1_main_ds = load_exact_tokens_iterable(args.main_data, args.token_budget, max_len=args.max_length)
    s1_aux_ds = load_exact_tokens_iterable(args.aux_data, aux_budget, max_len=args.max_length, cycle=True)
    s1_aux_ds, control_spec = wrap_aux_dataset_for_control(s1_aux_ds, args)
    if args.aux_task or args.aux_mode or args.aux_level:
        print(f"🧪 CONTROL = {control_spec}")
    if args.train_source_loss:
        s1_main_ds, s1_aux_ds = add_source(s1_main_ds, 0), add_source(s1_aux_ds, 1)

    parts = [d for d in [s1_main_ds, s1_aux_ds] if d is not None]
    if len(parts) == 2:
        p_main = 1 / (1 + args.aux_ratio)
        s1_final = interleave_datasets(
            parts, probabilities=[p_main, 1 - p_main],
            seed=42, stopping_strategy="first_exhausted",
        ).shuffle(seed=42, buffer_size=100)
    else:
        s1_final = parts[0].shuffle(seed=42, buffer_size=100)
    if args.train_source_loss:
        s1_final = pack_by_source(s1_final, tokenizer, args.max_length)

    s2_main_ds = None  # disabled in iterable mode
else:
    s1_main_ds, main_rows_consumed = load_exact_tokens_materialized(
        args.main_data, args.token_budget, stream_skip=0, max_len=args.max_length)
    s1_aux_ds, _ = load_exact_tokens_materialized(
        args.aux_data, aux_budget, stream_skip=0, max_len=args.max_length)
    if (args.aux_task or args.aux_mode or args.aux_level) and s1_aux_ds is not None:
        s1_aux_ds = s1_aux_ds.filter(
            lambda row: row_filter(row, task=args.aux_task, mode=args.aux_mode, level=args.aux_level)
        )
    s1_final = concatenate_datasets([d for d in [s1_main_ds, s1_aux_ds] if d]).shuffle(seed=42)

    s2_budget = int(args.token_budget * args.phase_2_ratio)
    s2_main_ds, _ = load_exact_tokens_materialized(
        args.main_data, s2_budget, stream_skip=main_rows_consumed, max_len=args.max_length)


# --- callback ---
class ScheduleFreeModeCallback(TrainerCallback):
    def __init__(self, optimizer): self.optimizer = optimizer
    def on_train_begin(self, args, state, control, **kwargs): self.optimizer.train()
    def on_step_end(self, args, state, control, **kwargs):
        if control.should_evaluate or control.should_save: self.optimizer.eval()
        else: self.optimizer.train()
    def on_train_end(self, args, state, control, **kwargs): self.optimizer.eval()


def downstream_eval(model, tokenizer):
    from reasoning_core.downstream_eval import run_harness, run_platinum

    return {
        **run_harness(model, tokenizer),
        **run_platinum(model, tokenizer),
    }


def configure_downstream_eval_metrics():
    wandb.define_metric("downstream_eval/index")
    wandb.define_metric("downstream_eval/score/*", step_metric="downstream_eval/index")
    wandb.define_metric("compare/*")


def log_downstream_eval(model, tokenizer, bucket, n_eval, global_step=None, max_steps=None, ordinal=None):
    scores = downstream_eval(model, tokenizer)
    metrics = {
        **scores,
        **{f"downstream_eval/score/{k}": v for k, v in scores.items()},
        "downstream_eval/bucket": bucket,          # 1-based intended eval bucket
        "downstream_eval/index": bucket - 1,       # 0-based axis for W&B plots
        "downstream_eval/n_eval": n_eval,
    }
    if global_step is not None:
        metrics["downstream_eval/global_step"] = global_step
    if max_steps is not None:
        metrics["downstream_eval/max_steps"] = max_steps
    if global_step is not None and max_steps:
        metrics["downstream_eval/progress"] = global_step / max_steps
    if ordinal is not None:
        metrics["downstream_eval/ordinal"] = ordinal

    if global_step is None:
        wandb.log(metrics)
    else:
        wandb.log(metrics, step=global_step)
    for key, value in scores.items():
        if key in {"nll/folio/nll", "nll/leaderboard_bbh/nll"} or key.startswith("platinum/"):
            wandb.run.summary[f"final/{key}"] = value
            wandb.run.summary[f"compare/{key.replace('/', '_')}"] = value
    local_metrics_sink.record(
        {
            **scores,
            "bucket": bucket,
            "index": bucket - 1,
            "n_eval": n_eval,
            **({"ordinal": ordinal} if ordinal is not None else {}),
        },
        kind="downstream",
        global_step=global_step,
        max_steps=max_steps,
    )


class ScopeEvalLogCallback(TrainerCallback):
    def __init__(self, n_aux_tl=0):
        self.n_aux_tl = n_aux_tl
        self.step = None
        self.seen = set()
        self.rt = self.samples = self.steps = 0.0

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs: return
        loss_keys = [k for k in logs if k.endswith("_loss") and (k.startswith("eval_") or k.startswith("eval/"))]
        acc_key = next((k for k in ("eval_mean_token_accuracy", "eval/mean_token_accuracy") if k in logs), None)
        if not loss_keys: return
        prefix = loss_keys[0][:-5]
        if acc_key: logs[f"{prefix}_mean_token_accuracy"] = logs.pop(acc_key)
        if prefix.startswith(("eval_aux_tl/", "eval/aux_tl/")):
            if self.step != state.global_step:
                self.step, self.seen = state.global_step, set()
                self.rt = self.samples = self.steps = 0.0
            rt = logs.pop(f"{prefix}_runtime", None)
            sps = logs.pop(f"{prefix}_samples_per_second", None)
            tps = logs.pop(f"{prefix}_steps_per_second", None)
            if rt is not None:
                self.rt += rt
                if sps is not None: self.samples += rt * sps
                if tps is not None: self.steps += rt * tps
            self.seen.add(prefix)
            if self.n_aux_tl and len(self.seen) >= self.n_aux_tl and self.rt:
                logs["eval_aux_tl_runtime"] = self.rt
                logs["eval_aux_tl_samples_per_second"] = self.samples / self.rt
                logs["eval_aux_tl_steps_per_second"] = self.steps / self.rt
        if wandb.run is not None:
            for key, value in logs.items():
                if key in ("eval_main_loss", "final_main_loss"):
                    wandb.run.summary[f"compare/{key}"] = value
                if key.endswith("_mean_token_accuracy") and key.startswith(("eval_aux_tl/", "eval_aux_tl_")):
                    wandb.run.summary[f"compare/{key.replace('/', '_')}"] = value

class DownstreamEvalCallback(TrainerCallback):
    """Run downstream eval at n-1 evenly-spaced buckets; final eval stays in existing block."""
    def __init__(self, model, tokenizer, optimizer, n=10):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.n = max(1, int(n))
        self.fired = set()
        self.ordinal = 0

    def buckets(self, max_steps):
        return {
            b: max(1, round(b * max_steps / self.n))
            for b in range(1, self.n)
        }

    def on_train_begin(self, args, state, control, **kwargs):
        if state.max_steps > 0:
            self.fired = {
                b for b, step in self.buckets(state.max_steps).items()
                if state.global_step >= step
            }

    def on_step_end(self, args, state, control, **kwargs):
        if state.max_steps <= 0:
            return

        due = [
            b for b, step in self.buckets(state.max_steps).items()
            if state.global_step >= step and b not in self.fired
        ]
        if not due:
            return

        b = max(due)
        self.fired.update(due)

        was_training = self.model.training
        self.optimizer.eval()
        self.model.eval()

        with torch.inference_mode():
            log_downstream_eval(
                self.model, self.tokenizer,
                bucket=b, n_eval=self.n,
                global_step=state.global_step,
                max_steps=state.max_steps,
                ordinal=self.ordinal,
            )
        self.ordinal += 1

        if was_training:
            self.model.train()

        if control.should_evaluate or control.should_save:
            self.optimizer.eval()
        else:
            self.optimizer.train()


class PeriodicAuxEvalCallback(TrainerCallback):
    """Run costly task/level evals at a lower cadence than main eval."""
    def __init__(self, eval_dataset, eval_steps, optimizer):
        self.eval_dataset = eval_dataset
        self.eval_steps = max(1, int(eval_steps))
        self.optimizer = optimizer
        self.trainer = None
        self.last_eval_step = None

    def attach_trainer(self, trainer):
        self.trainer = trainer
        prep_args = copy.copy(trainer.args)
        prep_args.dataset_num_proc = None
        packing = prep_args.packing
        if getattr(prep_args, "eval_packing", None) is not None:
            packing = prep_args.eval_packing
        self.eval_dataset = {
            key: trainer._prepare_dataset(dataset, trainer.processing_class, prep_args, packing, None, key)
            for key, dataset in self.eval_dataset.items()
        }
        trainer.aux_tl_eval_dataset = self.eval_dataset

    def on_step_end(self, args, state, control, **kwargs):
        if not self.eval_dataset or self.trainer is None or state.global_step <= 0:
            return
        if state.global_step % self.eval_steps:
            return
        if self.last_eval_step == state.global_step:
            return

        self.last_eval_step = state.global_step
        self.optimizer.eval()
        self.trainer.evaluate(eval_dataset=self.eval_dataset, metric_key_prefix="eval")
        if control.should_evaluate or control.should_save:
            self.optimizer.eval()
        else:
            self.optimizer.train()


common_args = dict(
    learning_rate=1.0, per_device_train_batch_size=per_device_bs,
    gradient_accumulation_steps=grad_accum, max_grad_norm=0.0, #grad clip is  handled by prodigy  
    logging_steps=15, gradient_checkpointing=args.gradient_checkpointing, bf16=True, report_to="wandb",
    eval_strategy="steps", packing=True,
    dataset_num_proc=1, max_length=args.max_length,
    save_strategy="steps", save_steps=args.save_steps, save_total_limit=args.save_total_limit,
)
if args.train_source_loss:
    common_args["packing"] = False
    common_args["remove_unused_columns"] = False

# Iterable mode: compute max_steps from token budget (trainer can't infer from a stream)
if args.iterable_mode:
    total_tokens = args.token_budget * (1 + args.aux_ratio)
    max_steps_s1 = max(1, int(total_tokens // (args.max_length * eff_batch)))
    print(f"📐 max_steps (stage1) = {max_steps_s1}  (eff_batch={eff_batch}, seq_len={args.max_length})")

MAIN_EVAL_ROUNDS = 30
AUX_EVAL_ROUNDS = 10
if args.iterable_mode:
    main_eval_steps = max(8, max_steps_s1 // MAIN_EVAL_ROUNDS)
    common_args["eval_steps"] = main_eval_steps
    aux_eval_steps = max(8, max_steps_s1 // AUX_EVAL_ROUNDS)
else:
    aux_eval_steps = None
opt_state = {"optimizer": None, "scheduler": None}


def run_stage(ds, stage_name, epochs=1.0, max_steps=-1, restart=False):
    if stage_name in completed:
        print(f"⏭️  Skipping {stage_name} (already completed)")
        return None
    if not ds: return None

    args.stage_name = stage_name
    run_id = f"{run_hash}-{stage_name}"
    wandb.init(project="rc-sft", group=group_id, id=run_id,
               name=f"{run_name}/{stage_name}", config=args, resume="allow", reinit=True)
    wandb.config.update({
        "compare/main_data": args.main_data,
        "compare/aux_task": args.aux_task or "all",
        "compare/aux_mode": args.aux_mode or "all",
        "compare/aux_level": args.aux_level or "all",
        "compare/aux_ratio": args.aux_ratio,
        "compare/seed": args.seed,
        "compare/from_scratch": args.from_scratch,
    }, allow_val_change=True)
    configure_downstream_eval_metrics()

    if restart or opt_state["optimizer"] is None:
        optimizer, scheduler = create_optimizer_and_scheduler(model, args)
        opt_state["optimizer"] = optimizer
        opt_state["scheduler"] = scheduler
    optimizer, scheduler = opt_state["optimizer"], opt_state["scheduler"]

    stage_dir = ckpt_dir / stage_name
    resume = stage_dir.exists() and any(stage_dir.glob("checkpoint-*"))
    if resume:
        print(f"♻️  Resuming {stage_name} from checkpoint in {stage_dir}")

    sft_kwargs = dict(output_dir=str(stage_dir), completion_only_loss=True, **common_args)
    if max_steps > 0:
        sft_kwargs["max_steps"] = max_steps
    else:
        sft_kwargs["num_train_epochs"] = epochs




    callbacks = [
        LocalMetricsCallback(local_metrics_sink),
        ScheduleFreeModeCallback(optimizer),
        DownstreamEvalCallback(model, tokenizer, optimizer, n=args.n_eval),
    ]
    aux_eval_callback = None
    if args.iterable_mode:
        aux_eval_callback = PeriodicAuxEvalCallback(aux_tl_evals, aux_eval_steps, optimizer)
        callbacks.append(aux_eval_callback)

    trainer_cls = trainer_cls_for_optimizer(args)
    trainer = trainer_cls(
        model=model, processing_class=tokenizer,
        args=SFTConfig(**sft_kwargs),
        train_dataset=ds, eval_dataset=trainer_evals,
        data_collator=SourceDataCollator(tokenizer.pad_token_id, max_length=args.max_length) if args.train_source_loss else None,
        optimizers=(optimizer, scheduler),
        callbacks=callbacks,
    )
    if aux_eval_callback is not None:
        aux_eval_callback.attach_trainer(trainer)

    trainer.callback_handler.callbacks.insert(
    0, ScopeEvalLogCallback(n_aux_tl=len(aux_tl_evals)) )

    optimizer.train()
    trainer.train(resume_from_checkpoint=resume)
    completed.add(stage_name)
    save_ckpt({"hash": run_hash, "group_id": group_id,
               "args": {k: str(v) for k, v in vars(args).items()},
               "completed_stages": sorted(completed), "done": False}, ckpt_file)
    return trainer


# --- 🚀 Execution ---
print(f"🚀 STAGE 1: Mixed ({args.main_data} + {args.aux_data})")
if args.iterable_mode:
    trainer = run_stage(s1_final, "stage1", max_steps=max_steps_s1)
else:
    trainer = run_stage(s1_final, "stage1")

if s2_main_ds:
    print(f"\n🚀 STAGE 2: Main Only ({args.main_data})")
    trainer = run_stage(s2_main_ds, "stage2")

if "eval" not in completed:
    if opt_state["optimizer"] is not None:
        opt_state["optimizer"].eval()

    if wandb.run is None:  # all stages skipped on resume
        wandb.init(project="rc-sft", group=group_id, id=f"{run_hash}-eval",
                   name=f"{run_name}/eval", config=args, resume="allow")
        configure_downstream_eval_metrics()

    final_step = trainer.state.global_step if trainer is not None else None
    final_max_steps = trainer.state.max_steps if trainer is not None else None
    args.stage_name = "eval"
    if trainer is None:
        local_metrics_sink.write_meta(max_steps=final_max_steps)
    model.eval()
    with torch.inference_mode():
        log_downstream_eval(
            model, tokenizer,
            bucket=args.n_eval, n_eval=args.n_eval,
            global_step=final_step,
            max_steps=final_max_steps,
        )

    if trainer is not None:
        trainer.evaluate(metric_key_prefix="final")
        if args.iterable_mode and aux_tl_evals:
            trainer.evaluate(eval_dataset=getattr(trainer, "aux_tl_eval_dataset", aux_tl_evals), metric_key_prefix="final")
    log_intrinsic_task_rewards(
        model, tokenizer, intrinsic_eval_splits, local_metrics_sink,
        global_step=final_step, max_steps=final_max_steps,
    )
    wandb.finish()
    save_ckpt({"hash": run_hash, "group_id": group_id,
               "args": {k: str(v) for k, v in vars(args).items()},
               "completed_stages": sorted(completed | {"eval"}), "done": True}, ckpt_file)

print('DONE ✨')
