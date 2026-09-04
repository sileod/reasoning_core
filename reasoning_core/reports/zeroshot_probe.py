#!/usr/bin/env python3
"""Zero-shot solve rate per task PER LEVEL, from a free API, as a reproducible difficulty probe.

Level coverage proves a task emits rows at each level, and prompt length proves the rungs differ in
size. Neither shows the rungs differ in DIFFICULTY. A solve rate that falls as the level rises is
that evidence; a flat one means the knob makes problems longer, not harder.

Several models, because a probe is only informative where the model can fail. One that solves
every rung says nothing about which rung is harder, so pick the weakest that still complies:

    ministral      Ministral-3-8B      free (albert), fastest, fails soonest
    mistral-small  Mistral-Small-24B   free (albert)
    v4flash        deepseek-v4-flash   free (albert), a reasoner held to no reasoning at all
    nim            llama-3.3-70b       the original pin; strongest, so it saturates the easy end

All are instructed to answer with no working, which is what makes the number reproducible: a
model left to think spends a variable, unbounded budget, so its solve rate records how long it
happened to think rather than how hard the task is.

Credentials come from the provider's env var (never inline a key):

    export NVIDIA_NIM_API_KEY=$(cat ~/.nvapi_key)
    python -m reasoning_core.reports.zeroshot_probe --levels 0 3 6 --n 2
    python -m reasoning_core.reports.zeroshot_probe --provider ministral --levels 0 3 6

Cache is keyed "task|level|model" -- with the model in the key, because two providers disagreeing
about a task is the signal, and a shared key would silently overwrite one with the other. A cell is
final only if it is ok AND was measured at the task's current `behavior_hash`, so editing a
generator re-opens exactly its own cells and resuming retries throttled ones for free. The free
tiers throttle hard: keep --workers at 1-2 and make repeated passes.
"""
from __future__ import annotations
import argparse, json, math, os, pathlib, random, sys, threading, time, urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor

TERSE = ("Answer the question. Reply with ONLY the final answer, no working, no punctuation "
         "beyond what the answer needs.")
# Tags make "did it comply" a fact we can read off the reply instead of a guess, so a reply we
# cannot parse is never mistaken for a hard task.
TAGGED = "Answer the question. Put ONLY the final answer between <answer> and </answer> tags."
# Only for a reasoner. Left alone it spends a variable, unbounded thinking budget, so its solve
# rate says how long it happened to think -- useless for calibration. Measured on albert, the API
# knobs for this (`reasoning_effort` low/minimal/none, `chat_template_kwargs.thinking`,
# `thinking.type`) are all accepted and all inert; the prompt cut 191 tokens to 8.
DIRECT = ("Answer immediately, from recognition alone. You are in a hurry: do NOT reason, do NOT "
          "work through steps, do NOT explain, do NOT check your answer. Your very first token "
          "must be <answer>, then the final answer, then </answer>. Nothing else.")

ALBERT = "https://albert.api.etalab.gouv.fr/v1/chat/completions"

PROVIDERS = {
    "nim": dict(url="https://integrate.api.nvidia.com/v1/chat/completions",
                model="meta/llama-3.3-70b-instruct",     # pinned: the signal is model-specific
                env="NVIDIA_NIM_API_KEY", key_file="~/.nvapi_key",
                system=TERSE, tagged=False, max_tokens=64),
    # Albert's three, weakest first. A probe wants a model that FAILS somewhere in the level
    # range: llama-70b and v4flash saturate the easy rungs, so the small ones carry more of the
    # gradient. Room for a short answer but far below what thinking needs -- an overrun is a
    # compliance failure we want to read as one, not a truncated ramble scored as a wrong answer.
    "ministral": dict(url=ALBERT, model="mistralai/Ministral-3-8B-Instruct-2512",
                      env="ALBERT_API_KEY", key_file=None,
                      system=TAGGED, tagged=True, max_tokens=256),
    "mistral-small": dict(url=ALBERT, model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
                          env="ALBERT_API_KEY", key_file=None,
                          system=TAGGED, tagged=True, max_tokens=256),
    "v4flash": dict(url=ALBERT, model="deepseek-v4-flash",
                    env="ALBERT_API_KEY", key_file=None,
                    system=DIRECT, tagged=True, max_tokens=256),
}
DEFAULT = "nim"


def key(p):
    k = os.environ.get(p["env"]) or ""
    if not k and p["key_file"]:
        f = pathlib.Path(p["key_file"]).expanduser()
        k = f.read_text().strip() if f.exists() else ""
    if not k:
        sys.exit(f"no key: set {p['env']}" + (f" or create {p['key_file']}" if p["key_file"] else ""))
    return k


def _ask(prompt, p, k, max_tokens, timeout):
    body = json.dumps({"model": p["model"], "temperature": 0, "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": p["system"]},
                                    {"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(p["url"], data=body, headers={
        "Authorization": f"Bearer {k}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    # `or ""`, not a bare index: a reasoner that spends its budget thinking returns content=null
    # with the thinking in `reasoning_content`. That used to raise, be caught, and score the cell
    # 0.0 -- the probe's own non-compliance filed as the task being hard. Now it reads as empty,
    # which answer_of reports as a format failure, which is what it is.
    return (d["choices"][0]["message"].get("content") or "").strip()


def ask(prompt, p, k, max_tokens=None, timeout=120, tries=4):
    """Backoff: a 429 is a property of the free tier, not of the task."""
    for i in range(tries):
        try:
            return _ask(prompt, p, k, max_tokens or p["max_tokens"], timeout)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i + random.random())


def answer_of(reply, p):
    """The answer to score, or None if the model did not comply with the reply format."""
    if p["tagged"]:
        return reply.split("<answer>")[-1].split("</answer>")[0].strip() if "</answer>" in reply else None
    return reply or None


def score_of(t, answer, x):
    """A score the caller can average. score_answer is task code and returns what it likes --
    arithmetics has been seen returning NaN -- and one NaN turns a whole solve rate into NaN."""
    try:
        s = float(t.score_answer(answer, x))
    except Exception:
        return 0.0
    return s if math.isfinite(s) else 0.0


def _measure(t, cases, p, k, base):
    """Score (prompt, entry) pairs. Format failures are counted, never averaged in as zeros:
    "the probe could not read the reply" and "the task is hard" are different findings."""
    scores, errs, unparsed = [], 0, 0
    for prompt, x in cases:
        try:
            reply = ask(prompt, p, k)
        except Exception:
            errs += 1
            continue
        answer = answer_of(reply, p)
        if answer is None:
            unparsed += 1
            continue
        scores.append(score_of(t, answer, x))
    asked = len(scores) + unparsed
    if not asked:
        return {**base, "status": "api-error", "n": 0, "errors": errs}
    if not scores:
        # Nothing was read, so nothing was measured. Not "ok" with a null rate, which the cache
        # would treat as final and never retry.
        return {**base, "status": "format-fail", "n": 0, "errors": errs, "format_ok": 0.0}
    return {**base, "status": "ok", "n": len(scores), "errors": errs,
            "format_ok": len(scores) / asked, "solve_rate": sum(scores) / len(scores)}


# Generation is serialized, API calls are not -- and the time is all in the API calls.
# Two reasons, both found the hard way at --workers 3: random.seed() is global, so parallel cells
# interleave on one RNG and --seed stops meaning anything (two models were ranked backwards from
# each other on the same seed until this held still); and z3-backed tasks hit an internal
# assertion and abort the whole process, taking every unwritten cell with them.
_GEN = threading.Lock()


def examples(task_name, n, level, seed):
    """n complete examples with prompts rendered, via the path the generation worker uses."""
    import reasoning_core as rc
    with _GEN:
        return _examples(rc, task_name, n, level, seed)


def _examples(rc, task_name, n, level, seed):
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


# A cell is prepared, then measured. The split is not cosmetic: preparing is where the task
# generator runs, and `template`'s runaway-generator deadline is a signal, so it only ever
# fires on the main thread. Run a generator in a worker and a task that never terminates
# never gets stopped -- it holds `_GEN` and the whole probe wedges, which is how a pass over
# 133 tasks spent three hours pinned at 90% CPU on `market_clearing` without writing a cell.
def prepare_rows(name, level, rows, p):
    """Score SHIPPED rows. Probing the data we actually ship makes the solve rate a statement about
    the dataset, not about re-running the generator today (and it needs no generator deps)."""
    import reasoning_core as rc
    from reasoning_core.template import Entry
    base = {"task": name, "level": level, "model": p["model"], "source": "shipped"}
    try:
        t = rc.get_task(name)
    except Exception as e:
        return None, {**base, "status": f"task:{type(e).__name__}", "n": 0}
    base["behavior_hash"] = t.behavior_hash()
    return (t, [(r["prompt"], Entry(r["metadata"], answer=r["answer"])) for r in rows], base), None


def prepare_cell(name, level, n, seed, p):
    """Draw ONE (task, level), on the main thread. Returns (ready, finished): one of them."""
    base = {"task": name, "level": level, "model": p["model"]}
    try:
        t, ex = examples(name, n, level, seed)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        # The deadline raises a BaseException on purpose, so that a generator cannot swallow
        # its own timeout. Here it is a verdict about the task like any other.
        return None, {**base, "status": f"gen:{type(e).__name__}", "n": 0}
    if not ex:
        return None, {**base, "status": "no-examples", "n": 0}
    base["behavior_hash"] = t.behavior_hash()
    return (t, ex, base), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=sorted(PROVIDERS), default=DEFAULT)
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
    p = PROVIDERS[a.provider]

    import reasoning_core as rc
    names = a.tasks or sorted(rc.list_tasks())
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else {}
    for flat in [x for x in list(done) if "|" not in x]:      # pre-multi-level cache was level 2
        done[f"{flat}|2"] = {**done.pop(flat), "level": 2}
    for old in [x for x in list(done) if x.count("|") == 1]:  # pre-multi-model cache was all NIM
        done[f"{old}|{PROVIDERS['nim']['model']}"] = {**done.pop(old), "model": PROVIDERS["nim"]["model"]}
    from reasoning_core.reports.task_staleness import live as live_hashes
    now = {k: v[0] for k, v in live_hashes(generated=True).items()}

    def final(t, lv):
        # Cells written before hashes were stamped keep their old meaning: absent == current.
        c = done.get(f"{t}|{lv}|{p['model']}", {})
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
    mine = [r for r in done.values() if r.get("model") == p["model"]]
    print(f"[zeroshot] {p['model']} levels {a.levels} n={a.n} -- {len(jobs)} cells to probe, "
          f"{sum(1 for r in mine if r.get('status') == 'ok')} ok")
    k = key(p)

    def record(r):
        done[f"{r['task']}|{r['level']}|{r['model']}"] = r
        out.write_text(json.dumps(done, indent=1, sort_keys=True))   # checkpoint every cell
        sr = f"{r['solve_rate']:.0%}" if r.get("solve_rate") is not None else r["status"]
        fmt = "" if r.get("format_ok", 1) == 1 else f"  format {r['format_ok']:.0%}"
        print(f"  {r['task']:32} L{r['level']}  {sr}{fmt}", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        prepare = ((lambda j: prepare_rows(j[0], j[1], cells[(j[0], j[1])][:a.n], p)) if cells
                   else (lambda j: prepare_cell(j[0], j[1], a.n, a.seed, p)))
        # One cell is drawn while the ones before it are still out with the provider, and the
        # window is bounded so the queue cannot run far ahead of what the workers can score.
        inflight = deque()
        for job in jobs:
            ready, finished = prepare(job)
            if finished is not None:
                record(finished)
                continue
            t, ex, base = ready
            inflight.append(pool.submit(_measure, t, ex, p, k, base))
            while len(inflight) > a.workers:
                record(inflight.popleft().result())
        while inflight:
            record(inflight.popleft().result())
    ok = sum(1 for r in done.values() if r.get("status") == "ok" and r.get("model") == p["model"])
    print(f"[zeroshot] {ok} cells ok for {p['model']} -> {out}")


if __name__ == "__main__":
    main()
