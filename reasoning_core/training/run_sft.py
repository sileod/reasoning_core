"""
for budget in 300_000_000 1_000_000_000; do   for r in 0.1 0.0 0.25 0.4 0.05; do     python run_sft.py --aux_ratio "$r" --token_budget "$budget";   done; done 
"""

import os, tempfile
from pathlib import Path
SAFE_TMP  = os.environ.get('SAFE_TMP',  str(Path.home() / '.cache'))
HF_CACHE = os.environ.get('HF_CACHE', str(Path.home() / '.cache' / 'huggingface'))
os.environ['HF_DATASETS_CACHE'] = HF_CACHE
for k in ('TMPDIR', 'TEMP', 'TMP'): os.environ[k] = SAFE_TMP
os.makedirs(SAFE_TMP, exist_ok=True)
tempfile.tempdir = SAFE_TMP



import warnings
warnings.filterwarnings("ignore")

import logging, os, argparse, torch, wandb, ast, json, hashlib, shutil
from faker import Faker
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Dataset, disable_caching
from transformers import AutoTokenizer, AutoModelForCausalLM, get_constant_schedule, TrainerCallback
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
from trl import SFTConfig, SFTTrainer
from tabulate import tabulate
from reasoning_core.downstream_eval import run_harness, run_platinum
from utils import ScheduleFreeModeCallback

#disable_caching()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_LOG_MODEL"] = "false" 
logging.getLogger("trl.trainer.sft_trainer").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)




# --- ⚙️ Config ---
parse_num = lambda s: int(float(s[:-1]) * {'K': 1e3, 'M': 1e6, 'B': 1e9}[s[-1].upper()]) if s[-1].isalpha() else int(s)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="PleIAs/Monad")
parser.add_argument('--token_budget', type=parse_num, default=10_000_000, help="Base token budget (Stage 1 Main)")
parser.add_argument('--aux_ratio', type=float, default=0.0, help="Ratio of Aux tokens relative to Main budget")
parser.add_argument('--phase_2_ratio', type=float, default=0.1, help="Ratio of Stage 2 Main tokens; 0 to skip")
parser.add_argument('--main_data', type=str, default="fw", choices=["fw", "synth","dolci"], help="Main dataset source")
parser.add_argument('--aux_data', type=str, default="rc", help="Auxiliary dataset source")
parser.add_argument('--max_length', type=int, default=1024)
parser.add_argument('--decay', type=float, default=0.01)
parser.add_argument('--from_scratch', type=ast.literal_eval, default=True)
parser.add_argument('--aux_version', type=str, default="rc10")
parser.add_argument('--script_version', type=str, default="4")
parser.add_argument('--aux_token', type=str, default="")


# Dataset Registry
DATA_MAP = {
    "fw": "HuggingFaceFW/fineweb-edu",
    "rc": "reasoning-core/symbolic-pretraining-pile",
    "synth": "tasksource/SYNTH",
    'dolci':'tasksource/dolci-instruct'
}

MODEL_MAP  = {
    "monad": "PleIAs/Monad", #56M
    "baguette" : "PleIAs/Baguettotron", #321M
    "h1": "tiiuae/Falcon-H1-Tiny-90M-Instruct",
    "ettin68":"jhu-clsp/ettin-decoder-68m",
    "ettin150":"jhu-clsp/ettin-decoder-150m"
}



def in_notebook():
    from IPython import get_ipython
    return getattr(get_ipython(), '__class__', None).__name__ == 'ZMQInteractiveShell'

args, _ = parser.parse_known_args() if in_notebook() else parser.parse_known_args()

if args.main_data=="dolci":
    args.from_scratch = False

args.model_name = MODEL_MAP.get(args.model_name, args.model_name)
SPECIAL = args.aux_token


print(tabulate(vars(args).items()))

# --- 💾 Checkpointing ---
def _args_hash(a):
    d = {k: str(v) for k, v in sorted(vars(a).items())}
    return hashlib.sha256(json.dumps(d).encode()).hexdigest()[:16]

def _run_name(seed_str):
    """Deterministic wandb-style name from a string seed."""
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
run_name = _run_name(run_hash)
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
    save_ckpt({"hash": run_hash, "group_id": group_id, "args": {k: str(v) for k, v in vars(args).items()}, "completed_stages": [], "done": False}, ckpt_file)

# --- 🧠 Model & Tokenizer ---

    
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32, #nb:torch_dtype is deprecated
    device_map="auto",
    attn_implementation="flash_attention_2",
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
if tokenizer.bos_token is None: 
    tokenizer.add_special_tokens({'bos_token': '<s>'})
    tokenizer.bos_token = '<s>'
    model.resize_token_embeddings(len(tokenizer))

if SPECIAL.strip():
    tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL]})
    model.resize_token_embeddings(len(tokenizer))

if args.from_scratch: model.apply(model._init_weights)
if getattr(tokenizer, "chat_template", None) is None:
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

# --- 📚 Smart Data Loading ---
def get_formatter(data_key):
    if data_key == "fw":
        return lambda x: {"prompt": "", "completion": x["text"] + tokenizer.eos_token}
    else:
        prefix = SPECIAL+"\n" if SPECIAL and data_key == "rc" else ""
        return lambda x: {"prompt": prefix+x["prompt"]+"\n", "completion": x.get("answer") + tokenizer.eos_token}

def load_exact_tokens(key, budget, stream_skip=0, max_len=None):
    if budget <= 0: 
        return None, 0
    
    hf_path = DATA_MAP.get(key, key)
    fmt_fn = get_formatter(key)
    
    print(f"⏳ Streaming {key} ({hf_path}) [skip={stream_skip}] target={budget/1e6:.2f}M tokens...")
    stream = load_dataset(hf_path, split="train", streaming=True).skip(stream_skip)
    tokens = rows = 0
    def gen():
        nonlocal tokens, rows
        for row in stream:
            rows += 1
            ex = fmt_fn(row)
            text = ex['prompt'] + ex['completion']
            length = len(tokenizer(text, add_special_tokens=True)['input_ids'])

            if max_len and length > max_len: 
                continue

            tokens += length
            yield ex
            if tokens >= budget: 
                break

    ds = Dataset.from_generator(gen)
    print(f"✅ Loaded {ds.num_rows} rows ({tokens} tokens).")
    return ds, rows

# --- 🧪 Experiment Setup ---

# 1. Load Stage 1: Main
s1_main_ds, main_rows_consumed = load_exact_tokens(
    args.main_data, args.token_budget, stream_skip=0, max_len=args.max_length
)

# 2. Load Stage 1: Aux
aux_budget = int(args.token_budget * args.aux_ratio)
s1_aux_ds, _ = load_exact_tokens(
    args.aux_data, aux_budget, stream_skip=0, max_len=args.max_length
)

# Merge Stage 1 (Main + Aux)
s1_final = concatenate_datasets([d for d in [s1_main_ds, s1_aux_ds] if d]).shuffle(seed=42)

# 3. Load Stage 2: Main (Continue from where S1 left off)
s2_budget = int(args.token_budget * args.phase_2_ratio)
s2_main_ds, _ = load_exact_tokens(
    args.main_data, s2_budget, stream_skip=main_rows_consumed, max_len=args.max_length
)

# 4. Load Evals (Fresh slices far into the datasets)
eval_main, _ = load_exact_tokens(args.main_data, 50_000, stream_skip=1_000_000, max_len=args.max_length)
eval_aux, _ = load_exact_tokens(args.aux_data, 50_000, stream_skip=100_000, max_len=args.max_length)

evals = {"main": eval_main}
if eval_aux: evals["aux"] = eval_aux

# --- 🔄 Training ---
m = 1
common_args = dict(
    learning_rate=1.0, per_device_train_batch_size=16//m,
    gradient_accumulation_steps=4*m, max_grad_norm=1.0,
    logging_steps=10, gradient_checkpointing=False, bf16=True, report_to="wandb",
    eval_strategy="steps", eval_steps=50, packing=True,
    dataset_num_proc=8, max_length=args.max_length,
    save_strategy="steps", save_steps=200, save_total_limit=2,
)

opt_state = {"optimizer": None, "scheduler": None}


def run_stage(ds, stage_name, epochs=1.0, restart=False):
    if stage_name in completed:
        print(f"⏭️  Skipping {stage_name} (already completed)")
        return
    args.stage_name = stage_name

    if not ds: return
    run_id = f"{run_hash}-{stage_name}"
    wandb.init(project="rc-sft", group=group_id, id=run_id, name=f"{run_name}/{stage_name}", config=args, resume="allow", reinit=True)
    if restart or opt_state["optimizer"] is None:
        optimizer = ProdigyPlusScheduleFree(model.parameters(), lr=1.0, weight_decay=args.decay,
            use_bias_correction=True, betas=(0.95, 0.99))
        opt_state["optimizer"], opt_state["scheduler"] = optimizer, get_constant_schedule(optimizer)
    optimizer, scheduler = opt_state["optimizer"], opt_state["scheduler"]

    stage_dir = ckpt_dir / stage_name
    resume = stage_dir.exists() and any(stage_dir.glob("checkpoint-*"))
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        args=SFTConfig(output_dir=str(stage_dir), num_train_epochs=epochs, completion_only_loss=True, **common_args),
        train_dataset=ds, eval_dataset=evals, optimizers=(optimizer, scheduler),
        callbacks=[ScheduleFreeModeCallback(optimizer)],
    )

    optimizer.train()
    trainer.train(resume_from_checkpoint=resume)
    completed.add(stage_name)
    save_ckpt({"hash": run_hash, "group_id": group_id, "args": {k: str(v) for k, v in vars(args).items()}, "completed_stages": sorted(completed), "done": False}, ckpt_file)

# --- 🚀 Execution ---
print(f"🚀 STAGE 1: Mixed ({args.main_data} + {args.aux_data})")
run_stage(s1_final, "stage1")

if s2_main_ds:
    print(f"\n🚀 STAGE 2: Main Only ({args.main_data})")
    run_stage(s2_main_ds, "stage2")

if "eval" not in completed:
    if opt_state["optimizer"] is not None:
        opt_state["optimizer"].eval()
    wandb.log(run_harness(model, tokenizer))
    wandb.log(run_platinum(model, tokenizer))
    wandb.finish()
    save_ckpt({"hash": run_hash, "group_id": group_id, "args": {k: str(v) for k, v in vars(args).items()}, "completed_stages": sorted(completed | {"eval"}), "done": True}, ckpt_file)

print('DONE ✨')
