#!/usr/bin/env python3
"""Build per-subject MMLU eval sets (BOTH cloze and normal/letter) for richer per-subject audit.

Splits the aggregate math/logic legs into their component MMLU subjects so transfer can be read PER SUBJECT
(and a broader math aggregate reconstructed) instead of leaning on high_school_math (270) only.
  * mmlu_<subj>_cloze_eval.jsonl : options OMITTED from prompt, answer = gold option TEXT  (format-fair NLL)
  * mmlu_<subj>_eval.jsonl       : options listed A./B./C./D. in prompt, answer = gold LETTER (standard MCQ)
Each leg gated EVAL_MMLU_<SUBJECT>[_CLOZE]=1. git-tracked; data_cache is per-machine, so run once per machine:
`python task_diagnostics/build_mmlu_subjects.py`.
"""
import json, string
from pathlib import Path
from datasets import load_dataset

SUBJECTS = ["abstract_algebra", "college_mathematics", "elementary_mathematics",
            "high_school_mathematics", "high_school_statistics", "formal_logic"]
MATH_MACRO = ["abstract_algebra", "college_mathematics", "elementary_mathematics",
              "high_school_mathematics", "high_school_statistics"]   # the 5 math subjects (NOT logic)
CAP_PER_SUBJ = 100   # balanced macro: equal rows/subject so the pooled leg ≈ subject-macro-average
DC = Path(__file__).resolve().parent.parent / "data_cache"
DC.mkdir(parents=True, exist_ok=True)

per_subj = {}   # subj -> (cloze_rows, normal_rows)
for subj in SUBJECTS:
    d = load_dataset("cais/mmlu", subj, split="test")
    cloze, normal = [], []
    for r in d:
        ch = [str(c).strip() for c in r["choices"]]
        gi = int(r["answer"])
        q = str(r["question"]).strip()
        if len(ch) < 2 or not (0 <= gi < len(ch)) or not ch[gi]:
            continue
        cloze.append({"prompt": f"{q}\nAnswer:", "answer": ch[gi], "choices": ch, "answer_idx": gi})
        menu = "\n".join(f"{string.ascii_uppercase[i]}. {c}" for i, c in enumerate(ch))
        normal.append({"prompt": f"{q}\n{menu}\nAnswer:", "answer": string.ascii_uppercase[gi]})
    per_subj[subj] = (cloze, normal)
    (DC / f"mmlu_{subj}_cloze_eval.jsonl").write_text("\n".join(map(json.dumps, cloze)) + "\n")
    (DC / f"mmlu_{subj}_eval.jsonl").write_text("\n".join(map(json.dumps, normal)) + "\n")
    print(f"{subj}: cloze n={len(cloze)}  normal n={len(normal)}")

# ── TRUE math macro leg (balanced over the 5 math subjects) ──────────────────────
# NB the legacy `mmlu_math_cloze`/`mmlu_math` legs are high_school_mathematics ONLY (270); this macro is
# the honest "MMLU-math". gated EVAL_MMLU_MATH_MACRO[_CLOZE]=1.
macro_cloze, macro_normal = [], []
for subj in MATH_MACRO:
    c, n = per_subj[subj]
    macro_cloze += c[:CAP_PER_SUBJ]; macro_normal += n[:CAP_PER_SUBJ]
(DC / "mmlu_math_macro_cloze_eval.jsonl").write_text("\n".join(map(json.dumps, macro_cloze)) + "\n")
(DC / "mmlu_math_macro_eval.jsonl").write_text("\n".join(map(json.dumps, macro_normal)) + "\n")
print(f"math_macro ({len(MATH_MACRO)} subj × ≤{CAP_PER_SUBJ}): cloze n={len(macro_cloze)}  normal n={len(macro_normal)}")
