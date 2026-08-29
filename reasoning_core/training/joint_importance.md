# Joint task importance

A single script (`joint_importance.py`) that scores each auxiliary task by how much it helps
**both** training regimes at once, so you can pick tasks that are good for a *pretraining mix*
rather than good in only one setting.

## What it measures

For each task it runs two influence measurements, **faithfully mirroring `run_sft.py`**:
same prompt/completion template (`Q: {prompt}\nA:` + ` {answer}`), completion-only loss,
SmolLM2-135M, and aux sampled at `p_aux = aux_ratio/(1+aux_ratio)` (run_sft sets
`p_main = 1/(1+aux_ratio)`, so `aux_ratio=0.2` ⇒ aux prob 1/6, not 1/5):

| regime | init | main data | steps |
|---|---|---|---|
| **fine-tuning** | pretrained | 80% dolci + 20% task | 300 |
| **pretraining** | from scratch | 80% fineweb + 20% task | 600 |

Influence = `ΔNLL` on held-out **BBH** = `NLL(main+task) − NLL(main-only baseline)`
(negative ⇒ the task *helps*). From scratch we also record the **fineweb-retention tax**
(`ΔNLL` on fineweb; positive ⇒ the task costs base-LM quality).

**`joint_z`** = mean of the per-task z-scores of `[finetune_BBH, pretrain_BBH, pretrain_FWtax]`
(each z-scored across the tasks scored together). **Lower = better in both regimes.** Averaged
over 2 seeds.

## Run

```bash
python joint_importance.py                              # ALL tasks in the source (default), 2 seeds
python joint_importance.py --tasks arithmetics,parsing  # a subset for a quick look
# options: --hf reasoning-core/procedural-pretraining-pile  --aux_ratio 0.2  --seeds 0,1  --n 1500
```

By default it scores **every task found in the HF source, sequentially** (single GPU), and is
**resumable** — results are written to `joint_importance.json` after every task, so a rerun
skips finished work and continues (important: a full ~40-task × 2-seed sweep is ~10 min/task/seed
≈ 13 h, long enough that a transient HF streaming hiccup would otherwise lose progress). Aux is
streamed from `--hf` and **deduplicated by prompt** (the pile/staging are not deduped). The
main-only baselines are computed once per regime/seed and cached. For a quick check, pass a
small `--tasks` subset.

## Why both regimes

Task usefulness is **regime-dependent**: across the rc tasks the fine-tuning and from-scratch
BBH rankings correlate only ρ≈0.2–0.5, and ~half the tasks flip helper↔hurter between them
(e.g. multi-step NLI helps fine-tuning but is a severe from-scratch hurter). So a single-regime
score is misleading for a pretraining mix — hence the joint score. The dominant from-scratch
cost is a **length/retention tax** (long, low-density answers hurt fineweb retention), while
fine-tuning rewards tasks that surface latent reasoning.

<!-- RESULTS -->
## Example (SmolLM2-135M, T300/T600, single seed; illustrative)

Lower `joint_z` = helps both regimes. Illustrative ranking from the project's influence runs
(single seed; pre-dates the exact run_sft template/ratio match above — the *ordering* is robust,
the absolute z's shift slightly). Re-running the script overwrites this section with 2-seed,
run_sft-faithful numbers.

| task                  |  joint_z |  ft_BBH |  pt_BBH |  pt_FWtax |
|-----------------------|---------:|--------:|--------:|----------:|
| navigation            |    -1.48 |  -0.287 |  -0.301 |    +0.034 |
| regex_reasoning       |    -1.35 |  -0.327 |  -0.182 |    +0.013 |
| arithmetics           |    -1.30 |  -0.279 |  -0.201 |    +0.005 |
| regex_following       |    -0.96 |  -0.247 |  -0.145 |    +0.001 |
| count_elements        |    -0.88 |  -0.350 |  -0.062 |    +0.019 |
| regex_induction       |    -0.86 |  -0.266 |  -0.164 |    +0.029 |
| coreference           |    -0.82 |  -0.337 |  -0.088 |    +0.031 |
| code_execution        |    -0.77 |  -0.242 |  -0.139 |    +0.016 |
| …                     |        … |       … |       … |         … |
| set_intersection      |    +0.74 |  -0.168 |  +0.030 |    +0.067 |
| parsing               |    +0.82 |  -0.191 |  +0.026 |    +0.087 |
| bayesian_intervention |    +0.90 |  -0.183 |  +0.048 |    +0.082 |
| bayesian_association  |    +0.91 |  -0.186 |  +0.053 |    +0.083 |
| diff_prediction       |    +0.92 |  -0.142 |  -0.043 |    +0.107 |
| planning              |    +1.15 |  -0.208 |  +0.067 |    +0.111 |
| table_conversion      |    +1.65 |  -0.031 |  -0.033 |    +0.130 |
| proof_reconstruction  |    +1.80 |  -0.057 |  +0.003 |    +0.141 |

Reading: `navigation`/`regex_*`/`arithmetics` help in both regimes; `proof_reconstruction`,
`table_conversion`, `planning` help fine-tuning weakly but carry the largest from-scratch
retention tax → poor pretraining-mix tasks.
