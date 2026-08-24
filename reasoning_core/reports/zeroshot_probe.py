#!/usr/bin/env python3
"""Zero-shot solve rate per task PER LEVEL, from a strong instruct model via the free NVIDIA NIM API.

Level coverage proves a task emits rows at each level, and prompt length proves the rungs differ in
size. Neither shows the rungs differ in DIFFICULTY. A solve rate that falls as the level rises is
that evidence; a flat one means the knob makes problems longer, not harder.

Credentials come from $NVIDIA_NIM_API_KEY, falling back to ~/.nvapi_key. Never inline a key.

    export NVIDIA_NIM_API_KEY=$(cat ~/.nvapi_key)
    python -m reasoning_core.reports.zeroshot_probe --levels 0 3 6 --n 2

Cache is keyed "task|level"; a cell is final only if it is ok AND was measured at the task's
current `behavior_hash`, so editing a generator re-opens exactly its own cells and resuming retries
throttled ones for free. The free tier throttles hard: keep --workers at 1-2 and make repeated passes.
"""
from __future__ import annotations
import argparse, json, os, pathlib, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.3-70b-instruct"          # pinned: the signal is model-specific
SYSTEM = ("Answer the question. Reply with ONLY the final answer, no working, no punctuation "
          "beyond what the answer needs.")


def key():
    k = os.environ.get("NVIDIA_NIM_API_KEY") or ""
    if not k:
        p = pathlib.Path("~/.nvapi_key").expanduser()
        k = p.read_text().strip() if p.exists() else ""
    if not k:
        sys.exit("no key: set NVIDIA_NIM_API_KEY or create ~/.nvapi_key")
    return k


def _ask(prompt, k, max_tokens, timeout):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {k}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def ask(prompt, k, max_tokens=64, timeout=90, tries=4):
    """Backoff: a 429 is a property of the free tier, not of the task."""
    for i in range(tries):
        try:
            return _ask(prompt, k, max_tokens, timeout)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i + random.random())


def examples(task_name, n, level, seed):
    """n complete examples with prompts rendered, via the path the generation worker uses."""
    import reasoning_core as rc
    random.seed(seed)
    t = rc.get_task(task_name)
    t.config.level = level
    out = []
    for x in t.generate_balanced_batch(batch_size=n, level=level) or []:
        prompt = getattr(x, "prompt", None)
        if prompt is None and hasattr(t, "render_prompt"):
            md = x.metadata if isinstance(x.metadata, dict) else json.loads(x.metadata)
            prompt = t.render_prompt(md)
        if prompt:
            out.append((prompt, x))
    return t, out


def probe_rows(name, level, rows, k):
    """Score SHIPPED rows. Probing the data we actually ship makes the solve rate a statement about
    the dataset, not about re-running the generator today (and it needs no generator deps)."""
    import reasoning_core as rc
    from reasoning_core.template import Entry
    base = {"task": name, "level": level, "source": "shipped"}
    try:
        t = rc.get_task(name)
    except Exception as e:
        return {**base, "status": f"task:{type(e).__name__}", "n": 0}
    base["behavior_hash"] = t.behavior_hash()
    scores, errs = [], 0
    for r in rows:
        try:
            reply = ask(r["prompt"], k)
        except Exception:
            errs += 1
            continue
        try:
            scores.append(float(t.score_answer(reply, Entry(r["metadata"], answer=r["answer"]))))
        except Exception:
            scores.append(0.0)
    if not scores:
        return {**base, "status": "api-error", "n": 0, "errors": errs}
    return {**base, "status": "ok", "n": len(scores), "errors": errs,
            "solve_rate": sum(scores) / len(scores)}


def probe_cell(name, level, n, seed, k):
    """Solve rate for ONE (task, level)."""
    base = {"task": name, "level": level}
    try:
        t, ex = examples(name, n, level, seed)
    except Exception as e:
        return {**base, "status": f"gen:{type(e).__name__}", "n": 0}
    if not ex:
        return {**base, "status": "no-examples", "n": 0}
    base["behavior_hash"] = t.behavior_hash()
    scores, errs = [], 0
    for prompt, x in ex:
        try:
            reply = ask(prompt, k)
        except Exception:
            errs += 1
            continue
        try:
            scores.append(float(t.score_answer(reply, x)))
        except Exception:
            scores.append(0.0)
    if not scores:
        return {**base, "status": "api-error", "n": 0, "errors": errs}
    return {**base, "status": "ok", "n": len(scores), "errors": errs,
            "solve_rate": sum(scores) / len(scores)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="examples per task per level")
    ap.add_argument("--levels", nargs="+", type=int, default=[0, 3, 6])
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=2, help="keep low; the free tier throttles")
    ap.add_argument("--out", default="reasoning_core/reports/build/zeroshot.json")
    ap.add_argument("--rows", default=None,
                    help="JSONL of shipped rows (task/level/prompt/answer/metadata); "
                         "score these instead of generating fresh examples")
    a = ap.parse_args()

    import reasoning_core as rc
    names = a.tasks or sorted(rc.list_tasks())
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else {}
    for flat in [x for x in list(done) if "|" not in x]:      # pre-multi-level cache was level 2
        done[f"{flat}|2"] = {**done.pop(flat), "level": 2}
    from reasoning_core.reports.task_staleness import live as live_hashes
    now = {k: v[0] for k, v in live_hashes().items()}

    def final(t, lv):
        # Cells written before hashes were stamped keep their old meaning: absent == current.
        c = done.get(f"{t}|{lv}", {})
        return c.get("status") == "ok" and c.get("behavior_hash", now.get(t)) == now.get(t)

    cells = None
    if a.rows:
        cells = {}
        for line in open(a.rows):
            r = json.loads(line)
            cells.setdefault((r["task"], r["level"]), []).append(r)
        names = sorted({t for t, _ in cells})
        a.levels = sorted({lv for _, lv in cells})
        jobs = [(t, lv) for (t, lv) in sorted(cells) if not final(t, lv)]
    else:
        jobs = [(t, lv) for t in names for lv in a.levels if not final(t, lv)]
    nok = sum(1 for r in done.values() if r.get("status") == "ok")
    print(f"[zeroshot] {MODEL} levels {a.levels} n={a.n} -- {len(jobs)} cells to probe, {nok} ok")
    k = key()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        run = ((lambda j: probe_rows(j[0], j[1], cells[(j[0], j[1])][:a.n], k)) if cells
               else (lambda j: probe_cell(j[0], j[1], a.n, a.seed, k)))
        for r in pool.map(run, jobs):
            done[f"{r['task']}|{r['level']}"] = r
            out.write_text(json.dumps(done, indent=1, sort_keys=True))   # checkpoint every cell
            sr = f"{r['solve_rate']:.0%}" if r.get("solve_rate") is not None else r["status"]
            print(f"  {r['task']:32} L{r['level']}  {sr}", flush=True)
    print(f"[zeroshot] {sum(1 for r in done.values() if r.get('status') == 'ok')}/{len(done)} "
          f"cells ok -> {out}")


if __name__ == "__main__":
    main()
