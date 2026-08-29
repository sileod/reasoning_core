"""joint_importance.py — per-task JOINT importance for a pretraining mix.

Scores each aux task by how much it helps BOTH training regimes at once, so you can pick
tasks that are good for a *pretraining mix* rather than good in only one setting:
  • fine-tuning   : SmolLM2-135M *pretrained*  + dolci   + task   (300 steps)
  • pretraining   : SmolLM2-135M *from scratch* + fineweb + task   (600 steps)
Influence = ΔNLL on held-out BBH = NLL(main+task) − NLL(main-only baseline); negative = helps.
We also track the from-scratch fineweb-retention tax (ΔNLL on fineweb). Per regime we z-score
Δ across tasks, then average the z's → one "joint_z" (lower = better in both regimes), 2 seeds.

Mirrors run_sft.py exactly: prompt/completion template `Q: {prompt}\nA:` + ` {answer}`,
completion-only loss, and aux sampled at p_aux = aux_ratio/(1+aux_ratio) (run_sft uses
p_main = 1/(1+aux_ratio)). Aux is streamed from a HF source and deduplicated by prompt.

By default it scores ALL tasks found in the HF source, sequentially, and is RESUMABLE
(writes after every task; rerun to continue / add tasks).

  python joint_importance.py                              # all tasks, 2 seeds
  python joint_importance.py --tasks arithmetics,parsing  # a subset
  python joint_importance.py --hf reasoning-core/procedural-pretraining-pile --aux_ratio 0.2 --seeds 0,1
"""
import os; os.environ.setdefault("HF_HUB_DISABLE_XET", "1")          # xet bridge stalls streaming
import argparse, json, itertools, statistics as st
from pathlib import Path
import torch
from datasets import load_dataset, IterableDataset, interleave_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from trl import SFTConfig, SFTTrainer
from tabulate import tabulate

MODEL = "HuggingFaceTB/SmolLM2-135M"; MAXLEN = 512; BATCH = 8; LR = 1e-4
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT  = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
REGIMES = [("finetune", "dolci", False, 300), ("pretrain", "fw", True, 600)]   # name, main, scratch, steps
BBH = ["boolean_expressions", "causal_judgement", "date_understanding", "formal_fallacies",
       "logical_deduction_five_objects", "multistep_arithmetic_two", "navigate", "object_counting",
       "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting", "sports_understanding"]

# run_sft.py template (get_formatter): QA → "Q: {prompt}\nA:" + " {answer}"; fw → raw text.
def fmt_qa(prompt, answer, eos): return {"prompt": f"Q: {prompt}\nA:", "completion": f" {answer}{eos}"}
def fmt_lm(text, eos):           return {"prompt": "", "completion": f"{text}{eos}"}
def cycle_gen(rows):
    def g():
        for r in itertools.cycle(rows): yield r
    return g

def bbh_eval(tok):  # held-out, scored in the same Q:/A: format the model is trained on
    out = []
    for s in BBH:
        try:
            for ex in list(load_dataset("lukaemon/bbh", s, split="test"))[5:20]:
                out.append((f"Q: {ex['input']}\nA:", f" {ex['target']}{tok.eos_token}"))
        except Exception: pass
    return out

def fw_eval(n=120):
    out = []
    for x in load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True).skip(800_000):
        if len(x["text"]) > 100: out.append(x["text"][:1500])
        if len(out) >= n: break
    return out

@torch.no_grad()
def nll_qa(model, tok, ex):
    model.eval(); tl = tn = 0.0
    for p, a in ex:
        pi = tok(p, add_special_tokens=False).input_ids; ai = tok(a, add_special_tokens=False).input_ids
        if len(pi) + len(ai) > MAXLEN or not ai: continue
        ids = torch.tensor([pi + ai], device=DEV); lab = ids.clone(); lab[0, :len(pi)] = -100
        o = model(ids, labels=lab); k = (lab[0, 1:] != -100).sum().item(); tl += o.loss.item() * k; tn += k
    return tl / max(tn, 1)

@torch.no_grad()
def nll_lm(model, tok, texts):
    model.eval(); tl = tn = 0.0
    for t in texts:
        ids = tok(t, add_special_tokens=False, truncation=True, max_length=MAXLEN, return_tensors="pt").input_ids.to(DEV)
        if ids.shape[1] < 2: continue
        o = model(ids, labels=ids); k = ids.shape[1] - 1; tl += o.loss.item() * k; tn += k
    return tl / max(tn, 1)

def main_gen(main, tok):
    eos = tok.eos_token
    if main == "dolci":
        def g():
            for x in load_dataset("tasksource/dolci-instruct", split="train", streaming=True):
                if x.get("answer"): yield fmt_qa(x["prompt"], x["answer"], eos)
    else:
        def g():
            for x in load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True):
                yield fmt_lm(x["text"][:1200], eos)
    return g

def list_tasks(hf, scan=60_000):
    seen = set()
    for i, x in enumerate(load_dataset(hf, split="train", streaming=True)):
        if i >= scan: break
        t = x.get("task") or ""
        if t and t != "reasoning_gym": seen.add(t)
    return sorted(seen)

def task_rows(hf, task, tok, n, scan_cap=400_000):
    """Deduplicated (by prompt) aux rows for one task."""
    eos = tok.eos_token; seen = set(); rows = []
    for i, x in enumerate(load_dataset(hf, split="train", streaming=True)):
        if i > scan_cap: break
        if (x.get("task") or "") != task or not x.get("answer"): continue
        p = x.get("prompt") or ""
        if p in seen: continue
        seen.add(p); rows.append(fmt_qa(p, x["answer"], eos))
        if len(rows) >= n: break
    return rows

def train(model, tok, ds, steps, seed):
    cfg = SFTConfig(output_dir="/tmp/joint_imp", overwrite_output_dir=True, max_steps=steps,
                    per_device_train_batch_size=BATCH, learning_rate=LR, weight_decay=0.01,
                    max_length=MAXLEN, packing=False, completion_only_loss=True,
                    bf16=(DT == torch.bfloat16), report_to="none", save_strategy="no",
                    logging_steps=10_000, seed=seed, disable_tqdm=True)
    SFTTrainer(model=model, processing_class=tok, train_dataset=ds, args=cfg).train()

def zscore(d):
    v = list(d.values()); m = st.mean(v); s = st.pstdev(v) or 1.0
    return {k: (x - m) / s for k, x in d.items()}

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="all", help="'all' (every task in the source) or a comma list")
    ap.add_argument("--hf", default="reasoning-core/procedural-pretraining-pile")
    ap.add_argument("--aux_ratio", type=float, default=0.2, help="aux:main token ratio (run_sft semantics)")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--n", type=int, default=1500, help="unique examples per task")
    ap.add_argument("--out", default=str(Path(__file__).parent / "joint_importance.json"))
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    p_aux = a.aux_ratio / (1 + a.aux_ratio)                          # run_sft: p_main = 1/(1+aux_ratio)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tasks = list_tasks(a.hf) if a.tasks == "all" else [t.strip() for t in a.tasks.split(",") if t.strip()]
    print(f"{len(tasks)} tasks, seeds {seeds}, p_aux={p_aux:.3f} ({a.hf})")
    bbh, fwe = bbh_eval(tok), fw_eval()
    R = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {"base": {}, "delta": {}}
    R.setdefault("base", {}); R.setdefault("delta", {})
    def save(): Path(a.out).write_text(json.dumps(R))

    for name, main, scratch, steps in REGIMES:
        mg = main_gen(main, tok)
        for seed in seeds:
            rk = f"{name}|{seed}"
            pending = [t for t in tasks if f"{rk}|{t}" not in R["delta"]]
            if not pending and rk in R["base"]:
                continue                                              # this regime/seed already complete
            torch.manual_seed(seed)
            model = (AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(MODEL), dtype=DT)
                     if scratch else AutoModelForCausalLM.from_pretrained(MODEL, dtype=DT)).to(DEV)
            init = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            reset = lambda: model.load_state_dict({k: v.to(DEV) for k, v in init.items()})
            if rk in R["base"]:
                b_bbh, b_fw = R["base"][rk]
            else:
                reset(); train(model, tok, IterableDataset.from_generator(mg), steps, seed)
                b_bbh, b_fw = nll_qa(model, tok, bbh), nll_lm(model, tok, fwe)
                R["base"][rk] = [b_bbh, b_fw]; save()
            for t in pending:
                rows = task_rows(a.hf, t, tok, a.n)
                if not rows:
                    print(f"  ! {t}: no rows (removed/renamed?), skipping"); R["delta"][f"{rk}|{t}"] = None; save(); continue
                reset()
                ds = interleave_datasets([IterableDataset.from_generator(mg),
                                          IterableDataset.from_generator(cycle_gen(rows))],
                                         probabilities=[1 - p_aux, p_aux], seed=seed, stopping_strategy="first_exhausted")
                train(model, tok, ds, steps, seed)
                R["delta"][f"{rk}|{t}"] = [nll_qa(model, tok, bbh) - b_bbh, nll_lm(model, tok, fwe) - b_fw]
                save(); print(f"  [{name} s{seed}] {t}: ΔBBH {R['delta'][f'{rk}|{t}'][0]:+.3f}")
            del model
            if DEV == "cuda": torch.cuda.empty_cache()

    # aggregate: mean Δ over seeds per regime → z-score → joint = mean of regime z's (lower = better)
    def mean_over_seeds(name, idx):
        out = {}
        for t in tasks:
            vals = [R["delta"][f"{name}|{s}|{t}"][idx] for s in seeds
                    if R["delta"].get(f"{name}|{s}|{t}")]
            if len(vals) == len(seeds): out[t] = st.mean(vals)
        return out
    comp = {"finetune_bbh": mean_over_seeds("finetune", 0),
            "pretrain_bbh": mean_over_seeds("pretrain", 0),
            "pretrain_fwtax": mean_over_seeds("pretrain", 1)}
    done = sorted(set.intersection(*[set(c) for c in comp.values()])) if all(comp.values()) else []
    if not done:
        print("no fully-scored tasks yet"); return
    Z = {k: zscore({t: comp[k][t] for t in done}) for k in comp}
    joint = {t: st.mean([Z[k][t] for k in Z]) for t in done}
    order = sorted(done, key=lambda t: joint[t])
    hdr = ["task", "joint_z", "ft_BBH", "pt_BBH", "pt_FWtax"]
    table = [[t, round(joint[t], 2), round(comp["finetune_bbh"][t], 3),
              round(comp["pretrain_bbh"][t], 3), round(comp["pretrain_fwtax"][t], 3)] for t in order]
    print("\n" + tabulate(table, headers=hdr, tablefmt="github"))
    R["joint"] = joint; save()
    md = Path(__file__).parent / "joint_importance.md"
    head = md.read_text().split("<!-- RESULTS -->")[0].rstrip() if md.exists() else "# Joint task importance\n"
    md.write_text(f"{head}\n\n<!-- RESULTS -->\n## Latest run (`{a.hf}`, seeds {seeds}, aux_ratio {a.aux_ratio}) "
                  f"— {len(done)} tasks, lower `joint_z` = helps both regimes\n\n"
                  + tabulate(table, headers=hdr, tablefmt="github") + "\n")
    print(f"\nwrote {a.out} and {md}")

if __name__ == "__main__":
    run()
