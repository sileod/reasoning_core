#!/usr/bin/env python3
"""Build the contrastive / preference eval legs of copyfree_battery_v5+ as 2-choice cloze rows.

PUBLIC entry point. The battery manifests in reasoning_core/training/ point at data_cache/*.jsonl
files. Frozen evaluation legs ship in resources/battery_legs.zip. This builder is
for creating new batteries from pinned sources, never refreshing an existing battery.

Equal-length truncation: both responses are cut to exactly T tokens, so length bias is gone by
construction rather than approximated by a ratio filter. Scoring is MEAN token logprob, so the two
choices are directly comparable. Prompt is capped from the TAIL -- in multi-turn rows the last user
turn is what the response answers.

    python scripts/build_preference_evals.py --split train --out helpsteer_correctness
"""
import argparse, json, os, random, re, sys
from pathlib import Path
# Resolve against the CWD, not the module: after moving into the package, parent.parent pointed
# at reasoning_core/data_cache. The battery manifests are loaded with an explicit data_dir too.
DC = Path(os.environ.get("EVAL_DATA_DIR", "data_cache"))
TOKM, TOKR = "HuggingFaceTB/SmolLM2-360M", "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"

REFUSAL = re.compile(r"^\s*(i'?m sorry|i am sorry|i cannot|i can'?t|i apolog|as an ai|i'?m not able)", re.I)
CODE = re.compile(r"```|\b(def |class |import |#include|SELECT .* FROM|function\s*\()", re.I)
HUB = Path("~/.cache/huggingface/hub").expanduser()

# Pinned source revisions. An unpinned build is NOT reproducible: `load_dataset` defaults to
# revision=None, which silently follows the dataset's moving HEAD, so two people running this a month
# apart get different rows and their eval numbers are not comparable.
REVISIONS = {
    "nvidia/HelpSteer2": "990b2711a36180dd19d9c94b8627844866f8982a",
    "trl-lib/ultrafeedback_binarized": "47124cb5778f5d50de1c7676a412828f3ea7c555",
    "tasksource/tasksource_dpo_pairs": "ebeb6a3f0160d90d6dc00b8d4e59ebcdaa7a5b2d",
    "allenai/llama-3.1-tulu-3-8b-preference-mixture": "78a6f00785946cd24276c5dd075f83a143a3b1e6",
    "aladinDJ/ultramix-DPO-annotated": "3fb2a270ec7c4756fa518b9452da2ddc967da2a2",
    "project-themis/Themis-CodePreference": "7c366b23590cc9ff8d372bb47280fcd474536344",
    "NextWealth/Python-DPO": "2e8512e71b138a2bef4dcd6c2c3fb8c160f09b99",
}


def _load(name, *args, **kw):
    """load_dataset pinned to the revision recorded above."""
    from datasets import load_dataset
    return load_dataset(name, *args, revision=REVISIONS.get(name), **kw)




def helpsteer2_pairs(a):
    """HelpSteer2 rows come in consecutive same-prompt pairs, each with 5 scored axes."""
    ds = _load("nvidia/HelpSteer2")[a.split]
    i = 0
    while i + 1 < len(ds):
        r0, r1 = ds[i], ds[i + 1]
        if r0["prompt"] != r1["prompt"]:
            i += 1
            continue
        i += 2
        yield r0, r1


def ultrafeedback_pairs(a):
    """UltraFeedback is already binarised: chosen/rejected conversations + GPT-4 scores 1-10.

    Re-shaped into the HelpSteer2 record form so one filter cascade serves both. `complexity` is
    absent, so it is set equal on both sides -- the |delta complexity| filter becomes a no-op rather
    than silently dropping everything.
    """
    import glob, pandas as pd
    pat = str(HUB / "datasets--trl-lib--ultrafeedback_binarized/snapshots/*/data"
              / ("test-*.parquet" if a.split != "train" else "train-*.parquet"))
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"no ultrafeedback parquet under {pat}")
    for f in files:
        for _, r in pd.read_parquet(f).iterrows():
            ch, rj = list(r["chosen"]), list(r["rejected"])
            if len(ch) < 2 or len(rj) < 2:
                continue
            prompt = "\n".join(m["content"] for m in ch[:-1] if m["role"] != "assistant")
            mk = lambda text, s: {"prompt": prompt, "response": text, a.axis: float(s),
                                  "complexity": 0}
            yield mk(ch[-1]["content"], r["score_chosen"]), mk(rj[-1]["content"], r["score_rejected"])


def tasksource_pairs(a):
    """tasksource/tasksource_dpo_pairs: 435 tasks, chosen/rejected are SHORT LABELS from one label set.

    This is structurally different from HelpSteer2/UltraFeedback and much closer to what actually
    works. The two candidates are drawn from the same label vocabulary ("neutral." vs
    "contradiction."), so they are matched in length, register and style by construction and differ
    only in which is correct -- the MINIMAL-PAIR property that made `balanced_copa` the one
    preference-style leg to survive the conditionality check. HS2 pairs differ in a dozen
    uncontrolled ways at once, so a base LM scores whichever difference it is most sensitive to
    (fluency); here there is no fluency difference to find.

    Prompts DO carry an option menu ("label with either X, Y or Z") and it is deliberately kept: the
    menu is required for the task to be answerable at all, and because it lists BOTH candidates
    symmetrically it shifts both NLLs together and cancels in the MARGIN. That is exactly what a
    margin is for. Read the margin, not the gold NLL.

    Stratified by task so no single one of the 435 dominates. Validation split by default -- 130547
    rows means the scarcity that forced HelpSteer2 onto train does not apply.
    """
    import collections
    d = _load("tasksource/tasksource_dpo_pairs", split=a.split)
    per, seen = a.per_task, collections.Counter()
    for r in d:
        t = str(r.get("task") or "?")
        if seen[t] >= per:
            continue
        ch, rj, pr = str(r["chosen"]).strip(), str(r["rejected"]).strip(), str(r["prompt"]).strip()
        if not ch or not rj or ch == rj or not pr:
            continue
        seen[t] += 1
        mk = lambda text, s: {"prompt": pr, "response": text, a.axis: float(s), "complexity": 0}
        yield mk(ch, 1.0), mk(rj, 0.0)


def tulu3_pairs(a):
    """allenai/llama-3.1-tulu-3-8b-preference-mixture: 272898 rows, conversation-shaped pairs.

    Same SHAPE as UltraFeedback (long free-form responses), so the pre-registered prediction is that
    it behaves like HS2/UF rather than like balanced_copa. There are no numeric quality scores, so
    the pair is binary: chosen=1, rejected=0 (use --min-delta 1). Stratified by `source` so no single
    upstream set dominates the 30-odd mixture components.
    """
    import collections
    d = _load("allenai/llama-3.1-tulu-3-8b-preference-mixture", split=a.split)
    per, seen = a.per_task, collections.Counter()
    for r in d:
        s = str(r.get("source") or "?")
        if seen[s] >= per:
            continue
        ch, rj = list(r["chosen"]), list(r["rejected"])
        if len(ch) < 2 or len(rj) < 2:
            continue
        seen[s] += 1
        prompt = str(r.get("prompt") or "")
        mk = lambda text, v: {"prompt": prompt, "response": text, a.axis: float(v), "complexity": 0}
        yield mk(ch[-1]["content"], 1.0), mk(rj[-1]["content"], 0.0)


def ultramix_pairs(a):
    """aladinDJ/ultramix-DPO-annotated: 190391 rows with CONTINUOUS reward scores + rich annotations.

    The valuable part is `chosen_instruct_reward` / `rejected_instruct_reward` (FsfairX-LLaMA3-RM):
    a real-valued quality gap per pair, so `--min-delta` selects the WIDEST-margin pairs rather than
    merely "chosen beats rejected". That makes this the strongest available test of the free-form
    preference class -- if the clearest pairs still read at chance, the class verdict is not a
    filtering artefact. Also filtered to English and stratified by `task_category`.
    """
    import collections
    d = _load("aladinDJ/ultramix-DPO-annotated", split=a.split)
    per, seen = a.per_task, collections.Counter()
    for r in d:
        if str(r.get("language") or "EN") != "EN":
            continue
        c = collections.Counter  # noqa: F841  (keep import local & cheap)
        t = str(r.get("task_category") or "?")
        if seen[t] >= per:
            continue
        try:
            cw, cl = float(r["chosen_instruct_reward"]), float(r["rejected_instruct_reward"])
        except (TypeError, ValueError):
            continue
        ch, rj = list(r["chosen"]), list(r["rejected"])
        if len(ch) < 2 or len(rj) < 2:
            continue
        seen[t] += 1
        prompt = str(r.get("prompt") or "")
        mk = lambda text, v: {"prompt": prompt, "response": text, a.axis: v, "complexity": 0}
        yield mk(ch[-1]["content"], cw), mk(rj[-1]["content"], cl)



def themis_code_pairs(a):
    """project-themis/Themis-CodePreference — Python subset, FUNCTIONAL-CORRECTNESS aspect.

    Codes are integers: language 1 = Python (sniffed 1463/1513), aspect 3 = functional correctness
    (sources MAGICODER_BUGGED + COMMITPREFS_FUNCTIONAL). ~70k such rows.

    Why this one is structurally promising where HelpSteer2 failed: chosen and rejected are the SAME
    program with a bug fixed — median similarity 0.939, 64% of pairs above 0.8. That is a genuine
    minimal pair on an OBJECTIVE axis (does it work), not a style judgement. The failure mode to
    avoid is truncation: the first differing token sits at median 40 but p90 190, so cutting to 96
    tokens makes 24% of pairs BYTE-IDENTICAL. Use short-turn FILTERING (--max-resp-tokens), never
    truncation, on this source.
    """
    d = _load("project-themis/Themis-CodePreference", split=a.split)
    for r in d:
        if r["language"] != 1 or r["aspect"] != 3:
            continue
        mk = lambda t, v: {"prompt": str(r["input"]), "response": str(t), a.axis: float(v),
                           "complexity": 0}
        yield mk(r["chosen"], 1.0), mk(r["rejected"], 0.0)


def python_dpo_pairs(a):
    """NextWealth/Python-DPO — 300 rows, chosen_code vs rejected_code-1 (per user: that column only).

    Small (n<=300, ~2.9pt floor) but short, self-contained functions on a correctness axis.
    """
    d = _load("NextWealth/Python-DPO", split=a.split)
    for r in d:
        ch, rj = str(r["chosen_code"]), str(r["rejected_code-1"])
        if not ch.strip() or not rj.strip() or ch == rj:
            continue
        mk = lambda t, v: {"prompt": str(r["instruction"]), "response": t, a.axis: float(v),
                           "complexity": 0}
        yield mk(ch, 1.0), mk(rj, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="helpsteer2",
                    choices=["helpsteer2", "ultrafeedback", "tasksource", "tulu3", "ultramix",
                             "themis_code", "python_dpo"])
    ap.add_argument("--per-task", type=int, default=6, help="tasksource: cap rows per task")
    ap.add_argument("--split", default="train")
    ap.add_argument("--axis", default="correctness")
    ap.add_argument("--min-delta", type=float, default=2)
    ap.add_argument("--max-complexity-delta", type=int, default=1)
    ap.add_argument("--trunc", type=int, default=128)
    ap.add_argument("--prompt-cap", type=int, default=0,
                    help="0 = derive from --max-length so prompt+answer always fits the battery; "
                         "a row that does not fit is scored inf and DROPPED SILENTLY by evaluate_mcq")
    ap.add_argument("--max-resp-tokens", type=int, default=0,
                    help="PREFER SHORT TURNS: keep only pairs whose responses are BOTH naturally "
                         "<= N tokens, complete and untruncated. Discards a lot; that is the point — "
                         "a truncated turn ends mid-sentence and is not a thing the model ever saw.")
    ap.add_argument("--min-resp-tokens", type=int, default=12,
                    help="floor, so a 'response' is not one word")
    ap.add_argument("--max-prompt-tokens", type=int, default=0,
                    help="PREFER SHORT TURNS: drop long prompts instead of tail-capping them "
                         "(a tail-cap throws the instruction away and keeps the trailing context)")
    ap.add_argument("--max-length", type=int, default=512,
                    help="must match the battery's max_length (copyfree_battery_*.json)")
    ap.add_argument("--drop-refusal", action="store_true")
    ap.add_argument("--drop-code", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--sentence-trunc", action="store_true",
                    help="cut at the last sentence boundary within --trunc instead of mid-word; "
                         "safe because scoring is MEAN token logprob, which normalises length")
    ap.add_argument("--train-format", action="store_true",
                    help="store the prompt with the trailing newline that data.py format_row adds, "
                         "so the eval prompt matches what the model is TRAINED on (prompt + chr(10))")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    DC.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKM, revision=TOKR)
    rng = random.Random(a.seed)
    if not a.prompt_cap:                       # leave headroom for the answer + a few join tokens
        a.prompt_cap = max(64, a.max_length - max(a.trunc, 0) - 16)
        print(f"  prompt-cap derived from max_length={a.max_length}: {a.prompt_cap} tokens")
    pairs = {"ultrafeedback": ultrafeedback_pairs, "tasksource": tasksource_pairs, "tulu3": tulu3_pairs, "ultramix": ultramix_pairs,
             "themis_code": themis_code_pairs, "python_dpo": python_dpo_pairs,
             "helpsteer2": helpsteer2_pairs}[a.source](a)

    n = {"pairs": 0, "delta": 0, "complexity": 0, "refusal": 0, "code": 0, "short": 0, "kept": 0}
    rows = []
    for r0, r1 in pairs:
        n["pairs"] += 1
        if abs(r0[a.axis] - r1[a.axis]) < a.min_delta:
            n["delta"] += 1
            continue
        if abs(r0["complexity"] - r1["complexity"]) > a.max_complexity_delta:
            n["complexity"] += 1
            continue
        win, lose = (r0, r1) if r0[a.axis] > r1[a.axis] else (r1, r0)
        if a.drop_refusal and (REFUSAL.search(win["response"]) or REFUSAL.search(lose["response"])):
            n["refusal"] += 1
            continue
        if a.drop_code and (CODE.search(r0["prompt"]) or CODE.search(win["response"])):
            n["code"] += 1
            continue
        wi = tok(win["response"], add_special_tokens=False).input_ids
        li = tok(lose["response"], add_special_tokens=False).input_ids
        if a.max_resp_tokens:          # short-turn mode: FILTER, never truncate
            if not (a.min_resp_tokens <= len(wi) <= a.max_resp_tokens
                    and a.min_resp_tokens <= len(li) <= a.max_resp_tokens):
                n["long_resp"] = n.get("long_resp", 0) + 1
                continue
        if a.max_prompt_tokens:
            if len(tok(r0["prompt"], add_special_tokens=False).input_ids) > a.max_prompt_tokens:
                n["long_prompt"] = n.get("long_prompt", 0) + 1
                continue
        if a.trunc > 0:
            if len(wi) < a.trunc or len(li) < a.trunc:
                n["short"] += 1
                continue
            w, l = tok.decode(wi[:a.trunc]), tok.decode(li[:a.trunc])
            if a.sentence_trunc:
                cut = lambda t: t[:m.end()] if (m := re.search(r"[.!?][\"')\]]?(?=\s|$)(?!.*[.!?])",
                                                              t, re.S)) else t
                w2, l2 = cut(w), cut(l)
                wt, lt = len(tok(w2, add_special_tokens=False).input_ids), \
                         len(tok(l2, add_special_tokens=False).input_ids)
                if wt >= 40 and lt >= 40:
                    w, l = w2, l2
                    n["sent_cut"] = n.get("sent_cut", 0) + 1
        else:
            # trunc<=0: keep responses whole. For short-label sources (tasksource) equal-length
            # truncation is meaningless and would chop "contradiction." to a fragment; the pair is
            # already length-matched because both labels come from one label set.
            w, l = win["response"], lose["response"]
        pi = tok(r0["prompt"], add_special_tokens=False).input_ids[-a.prompt_cap:]
        if w == l:        # truncation can collapse a real pair (Themis: 24% identical at T=96)
            n["degenerate"] = n.get("degenerate", 0) + 1
            continue
        gold = rng.randint(0, 1)                       # randomise position; scoring is per-choice
        ch = [w, l] if gold == 0 else [l, w]
        prompt_text = tok.decode(pi) + ("\n" if a.train_format else "")
        rows.append({"prompt": prompt_text, "answer": w, "choices": ch, "answer_idx": str(gold),
                     f"delta_{a.axis}": win[a.axis] - lose[a.axis]})
        n["kept"] += 1

    # A row longer than the battery's max_length is dropped silently at eval time. Refuse to ship one.
    over = [r for r in rows
            if len(tok(r["prompt"], add_special_tokens=False).input_ids)
            + max(len(tok(c, add_special_tokens=False).input_ids) for c in r["choices"]) > a.max_length]
    if over:
        print(f"  !! {len(over)}/{len(rows)} rows ({len(over)/len(rows):.1%}) exceed max_length="
              f"{a.max_length} and WOULD BE SILENTLY DROPPED by evaluate_mcq -- removing them")
        keep = {id(r) for r in over}
        rows = [r for r in rows if id(r) not in keep]
    rng.shuffle(rows)
    if a.limit:
        rows = rows[:a.limit]
    p = DC / f"{a.out}_eval.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{p}  n={len(rows)}")
    print("  cascade: " + "  ".join(f"{k}={v}" for k, v in n.items()))
    g = sum(int(r["answer_idx"]) for r in rows)
    print(f"  gold at idx1: {g}/{len(rows)}   verified answer==choices[idx]: "
          f"{sum(r['answer'] == r['choices'][int(r['answer_idx'])] for r in rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
