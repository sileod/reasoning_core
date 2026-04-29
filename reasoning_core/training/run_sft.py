"""
for budget in 300_000_000 1_000_000_000; do   for r in 0.1 0.0 0.25 0.4 0.05; do     python run_sft.py --aux_ratio "$r" --token_budget "$budget";   done; done
"""

import os, tempfile
from pathlib import Path

MEMORY = Path("/dev/shm/hf_cache")  # Use memory = Path.home() for disk

SAFE_TMP = os.environ.get('SAFE_TMP', str(MEMORY / '.cache'))
HF_CACHE = os.environ.get('HF_CACHE', str(MEMORY / '.cache' / 'huggingface'))
os.environ['HF_HOME'] = HF_CACHE
os.environ['HF_DATASETS_CACHE'] = HF_CACHE
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ['WANDB__SERVICE_WAIT']="120"
os.environ['WANDB_INIT_TIMEOUT']="200"
for k in ('TMPDIR', 'TEMP', 'TMP'): os.environ[k] = SAFE_TMP
os.makedirs(SAFE_TMP, exist_ok=True)
tempfile.tempdir = SAFE_TMP

from itertools import islice

import warnings
warnings.filterwarnings("ignore")

import logging, argparse, torch, wandb, ast, json, hashlib, shutil
from faker import Faker
from datasets import (
    load_dataset, concatenate_datasets, interleave_datasets,
    Dataset, IterableDataset, disable_caching,
)
from huggingface_hub import dataset_info
import hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM, get_constant_schedule, TrainerCallback, AutoConfig
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
from trl import SFTConfig, SFTTrainer
from tabulate import tabulate
from reasoning_core.downstream_eval import run_harness, run_platinum, run_bbh

disable_caching()
logging.getLogger("trl.trainer.sft_trainer").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


# --- ⚙️ Config ---
parse_num = lambda s: int(float(s[:-1]) * {'K': 1e3, 'M': 1e6, 'B': 1e9}[s[-1].upper()]) if s[-1].isalpha() else int(s)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="ettin68")
parser.add_argument('--token_budget', type=parse_num, default=100_000_000)
parser.add_argument('--aux_ratio', type=float, default=0.2)
parser.add_argument('--phase_2_ratio', type=float, default=0.0)
parser.add_argument('--main_data', type=str, default="dolci", choices=["fw", "synth", "dolci"])
parser.add_argument('--aux_data', type=str, default="rc")
parser.add_argument('--max_length', type=int, default=1024)
parser.add_argument('--decay', type=float, default=0.01)
parser.add_argument('--from_scratch', type=ast.literal_eval, default=True)
parser.add_argument('--aux_version', type=str, default="rc12")
parser.add_argument('--script_version', type=str, default="10")
parser.add_argument('--aux_token', type=str, default="")
parser.add_argument('--iterable_mode', type=ast.literal_eval, default=True)
parser.add_argument('--title', type=str, default=True)
parser.add_argument('--seed', type=int, default=True)

DATA_MAP = {
    "fw": "HuggingFaceFW/fineweb-edu",
    "rc": "reasoning-core/procedural-pretraining-pile",
    "synth": "tasksource/SYNTH",
    "dolci": "tasksource/dolci-instruct",
}

MODEL_MAP = {
    "monad": "PleIAs/Monad",
    "baguette": "PleIAs/Baguettotron",
    "h1": "tiiuae/Falcon-H1-Tiny-90M-Instruct",
    "ettin68": "jhu-clsp/ettin-decoder-68m",
    "ettin150": "jhu-clsp/ettin-decoder-150m",
}


def in_notebook():
    from IPython import get_ipython
    return getattr(get_ipython(), '__class__', None).__name__ == 'ZMQInteractiveShell'

args, _ = parser.parse_known_args()
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
    d = {k: str(v) for k, v in sorted(vars(a).items())}
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

if ckpt and ckpt.get("hash") == run_hash:
    if ckpt.get("done"):
        print("✅ Run already completed with these args. Exiting.")
        exit(0)
    completed = set(ckpt.get("completed_stages", []))
    group_id = ckpt["group_id"]
    print(f"♻️  Resuming {run_name} — completed stages: {completed}")
else:
    completed = set()
    group_id = f"{args.main_data}+{args.aux_data}-{run_name}"
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
    else:
        prefix = SPECIAL + "\n" if SPECIAL and data_key == "rc" else ""
        return lambda x: {"prompt": prefix + x["prompt"] + "\n",
                          "completion": x.get("answer") + tokenizer.eos_token}


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
def load_exact_tokens_iterable(key, budget, max_len=None, chars_per_token=4.0):
    if budget <= 0:
        return None

    hf_path = DATA_MAP.get(key, key)
    fmt_fn = get_formatter(key)
    print(f"⏳ [iterable] {key} ({hf_path}) infinite stream (budget handled by max_steps)")

    max_chars = max_len * chars_per_token if max_len else None

    def gen():
        stream = load_dataset(hf_path, split="train", streaming=True)
        for row in stream:
            ex = fmt_fn(row)
            if max_chars and (len(ex["prompt"]) + len(ex["completion"])) > max_chars:
                continue
            yield ex

    ds = IterableDataset.from_generator(gen)
    return ds


# --- 🧪 Experiment Setup ---

# Evals: always materialized (tiny, trainer needs __len__)
eval_main, _ = load_exact_tokens_materialized(args.main_data, 50_000, stream_skip=1_000_000, max_len=args.max_length)
eval_aux, _ = load_exact_tokens_materialized(args.aux_data, 50_000, stream_skip=100_000, max_len=args.max_length)
evals = {"main": eval_main}
if eval_aux: evals["aux"] = eval_aux

aux_budget = int(args.token_budget * args.aux_ratio)

if args.iterable_mode:
    s1_main_ds = load_exact_tokens_iterable(args.main_data, args.token_budget, max_len=args.max_length)
    s1_aux_ds = load_exact_tokens_iterable(args.aux_data, aux_budget, max_len=args.max_length)

    parts = [d for d in [s1_main_ds, s1_aux_ds] if d is not None]
    if len(parts) == 2:
        p_main = 1 / (1 + args.aux_ratio)
        s1_final = interleave_datasets(
            parts, probabilities=[p_main, 1 - p_main],
            seed=42, stopping_strategy="all_exhausted",
        ).shuffle(seed=42, buffer_size=100)
    else:
        s1_final = parts[0].shuffle(seed=42, buffer_size=100)

    s2_main_ds = None  # disabled in iterable mode
else:
    s1_main_ds, main_rows_consumed = load_exact_tokens_materialized(
        args.main_data, args.token_budget, stream_skip=0, max_len=args.max_length)
    s1_aux_ds, _ = load_exact_tokens_materialized(
        args.aux_data, aux_budget, stream_skip=0, max_len=args.max_length)
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


# --- 🔄 Training ---
_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
_model_params = sum(p.numel() for p in model.parameters()) / 1e6
available_memory_factor = max(1, min(16, round(2 * (_model_params / 68) * (24 / _total_gb))))

per_device_bs = 16 // available_memory_factor
grad_accum = 4 * available_memory_factor

common_args = dict(
    learning_rate=1.0, per_device_train_batch_size=per_device_bs,
    gradient_accumulation_steps=grad_accum, max_grad_norm=1.0,
    logging_steps=10, gradient_checkpointing=False, bf16=True, report_to="wandb",
    eval_strategy="steps", eval_steps=50, packing=True,
    dataset_num_proc=1, max_length=args.max_length,
    save_strategy="steps", save_steps=200, save_total_limit=2,
)

# Iterable mode: compute max_steps from token budget (trainer can't infer from a stream)
if args.iterable_mode:
    eff_batch = per_device_bs * grad_accum
    total_tokens = args.token_budget * (1 + args.aux_ratio)
    max_steps_s1 = max(1, int(total_tokens // (args.max_length * eff_batch)))
    print(f"📐 max_steps (stage1) = {max_steps_s1}  (eff_batch={eff_batch}, seq_len={args.max_length})")

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

    if restart or opt_state["optimizer"] is None:
        optimizer = ProdigyPlusScheduleFree(
            model.parameters(), lr=1.0, weight_decay=args.decay,
            use_bias_correction=True, betas=(0.95, 0.99))
        opt_state["optimizer"] = optimizer
        opt_state["scheduler"] = get_constant_schedule(optimizer)
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

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        args=SFTConfig(**sft_kwargs),
        train_dataset=ds, eval_dataset=evals,
        optimizers=(optimizer, scheduler),
        callbacks=[ScheduleFreeModeCallback(optimizer)],
    )

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
    wandb.log(run_harness(model, tokenizer))
    wandb.log(run_platinum(model, tokenizer))
    try: wandb.log(run_bbh(model, tokenizer))
    except Exception as e: print(e)
    if trainer is not None:
        trainer.evaluate(metric_key_prefix="final")
    wandb.finish()
    save_ckpt({"hash": run_hash, "group_id": group_id,
               "args": {k: str(v) for k, v in vars(args).items()},
               "completed_stages": sorted(completed | {"eval"}), "done": True}, ckpt_file)

print('DONE ✨')