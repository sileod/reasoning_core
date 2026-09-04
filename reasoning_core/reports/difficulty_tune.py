#!/usr/bin/env python3
"""Turn a measured difficulty curve into a proposed `apply_difficulty`, as a tool, not a gate.

`zeroshot_probe` measures solve rate per level. That number is most useful BEFORE anything is
promoted: a flat curve says the level knob makes problems longer rather than harder, and that is a
fixable defect in four lines of config, not grounds for rejecting a task. Rejection sampling on
this signal would be far too harsh -- it would throw away good tasks for a bad ladder.

So this reads the curve, says what is wrong with it, and asks a model for a replacement
`apply_difficulty`. It never edits the repo unless you pass --apply; by default it prints a diff.

    python -m reasoning_core.reports.zeroshot_probe --provider ministral --levels 0 2 4 6 --n 8
    python -m reasoning_core.reports.difficulty_tune --provider ministral

The model returns the new method, and difflib makes the diff -- not the other way round. A model
asked for a unified diff has to reproduce context lines byte-exactly, which it will eventually get
wrong in a way that applies cleanly and means something else.

Every proposal is verified before you see it, by compiling it and binding it to a subclass in
memory. A method that crashes, ignores `level`, stops the task generating, restates the declared
defaults instead of reading them, or computes the identical config at every level is reported as
rejected rather than printed as a suggestion. Each of those shipped a bad diff once already; the
one retry quotes the rejection back, which is usually enough.
"""
from __future__ import annotations
import argparse, difflib, inspect, json, os, pathlib, sys, textwrap

from ..task_search.wave_proposer import ChatClient

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_ENDPOINT = "https://albert.api.etalab.gouv.fr/v1/chat/completions"
DEFAULT_API_KEY_ENV = "ALBERT_API_KEY"   # ChatClient defaults to NVIDIA's key

# What a ladder should do: start solvable, end hard, and fall in between. The bands are wide
# because the probe is a few samples per cell -- this decides which tasks are worth a look, and
# the diff decides nothing at all until a person reads it.
EASY_FLOOR = 0.60     # the bottom level should be mostly solvable
HARD_CEIL = 0.30      # the top level should mostly not be
MIN_SPAN = 0.25       # below this the knob is not moving difficulty
MIN_N = 5             # a solve rate over 3 samples has a standard error near 0.3

SYSTEM = """You tune the difficulty ladder of procedurally generated reasoning tasks.

Each task has a Config whose `apply_difficulty(self, level)` sets the generation parameters for an
integer level from the base field values. A good ladder is solvable at level 0, hard at the top
level, and falls monotonically in between. You are given the measured zero-shot solve rate per
level and the current method, and you return a replacement.

Rules:
- Change only `apply_difficulty`. Never rename or remove config fields, and never introduce one.
- Keep it a pure non-recursive formula in `level`: level 3 must not depend on level 2.
- The framework RESETS every field to its declared default before calling you, so `self.x` is
  always the base value on entry. Do not re-assign base constants and do not guard on `level > 0`.
- Write every field relative to what it already holds -- `self.x = self.x + 2 * level`, never
  `self.x = 7 + 2 * level`. Hardcoding a default duplicates it, and the copy goes stale.
- Use ONLY names the task's module already has; the prompt lists the helpers available.
- Scale the knobs that make a problem structurally harder (depth, size, number of constraints,
  branching), not the ones that only make it longer to read or print.
- Level 0 must stay at or near the base values, so the easy rung stays easy.
- Prefer the smallest change that fixes the measured defect.

Output one JSON object, no prose, no markdown fence:
{"apply_difficulty": "<complete method source, def line included, 4-space base indent>",
 "rationale": "<one or two sentences on what you changed and why>"}"""


def curve(cache, model, min_n=MIN_N):
    """{task: (points, holes)} for one model, from the probe cache.

    `holes` are probed levels whose cell cannot be trusted, and a curve with holes is not
    diagnosed at all. The dropped cells are the HARD ones -- format failures and short batches
    cluster at the top of the ladder -- so dropping them per cell truncates the curve from the
    top and makes a falling task look flat. graph_pathfinding measured 53/53/25/0 and was
    diagnosed "flat" off the two surviving cells, which would have bought a patch it did not need.
    """
    points, holes = {}, {}
    for key, cell in cache.items():
        parts = key.split("|")
        if len(parts) != 3 or parts[2] != model:
            continue
        task, level = parts[0], int(parts[1])
        if cell.get("status") != "ok":
            why = cell.get("status")
        elif cell.get("format_ok", 1.0) < 0.5:
            why = f"format {cell['format_ok']:.0%}"
        elif cell.get("n", 0) < min_n:
            why = f"n={cell.get('n', 0)}"
        else:
            why = None
        if why:
            holes.setdefault(task, {})[level] = why
        else:
            points.setdefault(task, {})[level] = cell["solve_rate"]
    return {t: (dict(sorted(points.get(t, {}).items())), dict(sorted(holes.get(t, {}).items())))
            for t in set(points) | set(holes)}


def diagnose(points):
    """What is wrong with this ladder, or None if nothing is."""
    rates = list(points.values())
    lo, hi = rates[0], rates[-1]
    if hi > lo + MIN_SPAN:
        return ("inverted", "solve rate RISES with level: the knob is making problems easier")
    if min(rates) >= EASY_FLOOR:
        return ("too-easy", f"solved at every level (min {min(rates):.0%}): the top rung is not hard")
    if max(rates) <= HARD_CEIL:
        return ("too-hard", f"unsolved at every level (max {max(rates):.0%}): the bottom rung is not easy")
    if lo - hi < MIN_SPAN:
        return ("flat", f"spans only {lo - hi:+.0%} across the range: the knob adds size, not difficulty")
    return None


def config_source(task_name):
    """(config class, file, method source). The method must be the task's own, not inherited."""
    import reasoning_core as rc
    cfg = type(rc.get_task(task_name).config)
    if "apply_difficulty" not in cfg.__dict__:
        raise LookupError(f"{cfg.__name__} inherits apply_difficulty; nothing here to tune")
    return cfg, pathlib.Path(inspect.getsourcefile(cfg)), inspect.getsource(cfg.apply_difficulty)


def compile_method(cfg, source):
    """The proposed method as a callable, in its own module's namespace so its imports resolve."""
    namespace = dict(sys.modules[cfg.__module__].__dict__)
    exec(textwrap.dedent(source), namespace)
    return namespace["apply_difficulty"]


def hardcoded(patched, varying):
    """The field whose base default the method ignores, or None.

    `apply_difficulty` is called on a config already reset to its declared defaults, so it must
    read them rather than restate them. A method that assigns `self.max_count = 3 + level` passes
    every other check here and still silently ignores the dataclass field it duplicates. Bumping
    a default and watching level 0 fail to follow is the cheapest way to catch that.
    """
    for name in sorted(varying):
        base = patched().to_dict().get(name)
        if not isinstance(base, (int, float)) or isinstance(base, bool):
            continue
        probe = patched(**{name: base + 7})
        probe.set_level(0)
        if probe.to_dict().get(name) != base + 7:
            return name
    return None


def verify(task_name, cfg, method, levels):
    """Run the proposal without touching the repo. Returns (ok, note)."""
    import reasoning_core as rc
    patched = type(cfg.__name__, (cfg,), {"apply_difficulty": method})
    states = {}
    for level in levels:
        try:
            c = patched()
            c.set_level(level)
            states[level] = c.to_dict()
        except Exception as e:
            return False, f"level {level} raised {type(e).__name__}: {e}"
    varying = {k for k in states[levels[0]]
               if k != "level" and len({json.dumps(s[k], default=str) for s in states.values()}) > 1}
    if not varying:
        return False, "config is identical at every level: the method ignores `level`"
    before = {}
    for level in levels:
        c = cfg()
        c.set_level(level)
        before[level] = c.to_dict()
    if before == states:
        # Rewriting `x += level` as `x = self.x + level` passes every other check here and
        # changes nothing. Presented as a fix, with a rationale, it is worse than no patch.
        return False, "no behavioural change: identical config at every level"
    stale = hardcoded(patched, varying)
    if stale:
        return False, (f"`{stale}` is hardcoded: changing its declared default no longer moves "
                       f"level 0, so the field and the formula have separate copies of the value")
    try:
        t = rc.get_task(task_name)
        t.config = patched()
        t.config.set_level(levels[-1])
        if not (t.generate_balanced_batch(batch_size=1, level=levels[-1]) or []):
            return False, f"generates nothing at level {levels[-1]}"
    except Exception as e:
        return False, f"generation at level {levels[-1]} raised {type(e).__name__}: {e}"
    return True, "varies " + ", ".join(sorted(varying))


def propose(client, task_name, cfg, points, verdict, detail, method_source, rejected=None):
    measured = "\n".join(f"  level {lv}: {r:.0%} solved" for lv, r in points.items())
    helpers = [n for n in ("sround", "math", "random")
               if n in sys.modules[cfg.__module__].__dict__]
    fields = "\n".join(f"  {f}: {v!r}" for f, v in sorted(cfg().to_dict().items())
                       if f not in ("level", "seed", "size"))
    user = (f"Task: {task_name}\nConfig class: {cfg.__name__}\n\n"
            f"Base field values (level 0):\n{fields}\n\n"
            f"Measured zero-shot solve rate:\n{measured}\n\n"
            f"Diagnosis: {verdict} -- {detail}\n\n"
            f"Helpers available in this module: {', '.join(helpers) or 'none -- use plain arithmetic'}\n\n"
            f"Current method:\n{method_source}")
    if rejected:
        user += (f"\n\nYour previous attempt was REJECTED: {rejected[1]}\n"
                 f"It was:\n{rejected[0]}\nFix that and return the method again.")
    return client.json(f"difficulty:{task_name}", SYSTEM, user)


def splice(path, old_method, new_method):
    """(patched file text, unified diff), or (None, why not)."""
    before = path.read_text()
    if before.count(old_method) != 1:
        return None, "current method source is not uniquely locatable in the file"
    body = textwrap.indent(textwrap.dedent(new_method).strip("\n"), "    ") + "\n"
    after = before.replace(old_method, body)
    try:
        name = str(path.relative_to(pathlib.Path.cwd()))
    except ValueError:
        name = str(path)
    return after, "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                               fromfile="a/" + name, tofile="b/" + name))


def tune(task_name, points, client, levels):
    """Diagnose one task and, if it needs it, propose and verify a new method."""
    verdict = diagnose(points)
    if not verdict:
        return {"task": task_name, "verdict": "ok"}
    row = {"task": task_name, "verdict": verdict[0], "detail": verdict[1], "curve": points}
    if verdict[0] == "too-hard":
        # Level 0 is the declared field defaults by construction, so an unsolvable level 0 is a
        # statement about those defaults. apply_difficulty is called after they are restored and
        # cannot lower them; asking anyway buys a confabulated no-op.
        return {**row, "error": "level 0 is already unsolvable, so the base field defaults are "
                                "what is too hard -- outside what apply_difficulty can change"}
    try:
        cfg, path, method_source = config_source(task_name)
    except Exception as e:
        return {**row, "error": str(e)}
    rejected, reply, method = None, None, None
    for attempt in range(2):        # one retry, with the rejection quoted back: the checks below
        try:                        # say exactly what is wrong, and saying it is most of the fix
            reply = propose(client, task_name, cfg, points, *verdict, method_source, rejected)
            proposed = reply["apply_difficulty"]
            method = compile_method(cfg, proposed)
        except Exception as e:
            rejected = (str(reply and reply.get("apply_difficulty", "")),
                        f"{type(e).__name__}: {e}")
            continue
        ok, note = verify(task_name, cfg, method, levels)
        if ok:
            break
        rejected, method = (proposed, note), None
    if method is None:
        return {**row, "error": f"rejected after {attempt + 1} attempts: {rejected[1]}"}
    patched, text = splice(path, method_source, proposed)
    if patched is None:
        return {**row, "error": text}
    return {**row, "rationale": reply.get("rationale", ""), "verified": note,
            "path": str(path), "patched": patched, "diff": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zeroshot", default="reasoning_core/reports/build/zeroshot.json")
    ap.add_argument("--provider", default="ministral",
                    help="which probe model's curve to read (see zeroshot_probe.PROVIDERS)")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    ap.add_argument("--apply", action="store_true", help="write the verified patches to the repo")
    ap.add_argument("--out", default=None, help="also write the proposals as JSON")
    a = ap.parse_args()

    from .zeroshot_probe import PROVIDERS
    cache = pathlib.Path(a.zeroshot)
    if not cache.exists():
        sys.exit(f"no probe results at {cache}; run zeroshot_probe first")
    model = PROVIDERS[a.provider]["model"]
    curves = curve(json.loads(cache.read_text()), model, a.min_n)
    if a.tasks:
        curves = {t: v for t, v in curves.items() if t in a.tasks}
    if not curves:
        sys.exit(f"no cells for {model}; run zeroshot_probe --provider {a.provider} first")

    usable, holed = {}, {}
    for task, (points, holes) in curves.items():
        if holes or len(points) < 2:
            holed[task] = (points, holes)
        else:
            usable[task] = points
    needs = {t: p for t, p in usable.items() if diagnose(p)}
    print(f"[tune] {model}: {len(usable)} complete curves, {len(needs)} to tune, "
          f"{len(holed)} incomplete")
    for task, (points, holes) in sorted(holed.items()):
        got = " ".join(f"L{lv}={r:.0%}" for lv, r in points.items()) or "nothing usable"
        bad = " ".join(f"L{lv}:{why}" for lv, why in holes.items())
        print(f"  skipped {task:28} {got}   [{bad}] -- probe more samples per cell")
    if not needs:
        return
    client = ChatClient(model=a.model, endpoint=a.endpoint,
                        api_key=os.environ.get(a.api_key_env),
                        temperature=0.2, reasoning_effort=None)
    rows = []
    for name, points in sorted(needs.items()):
        levels = sorted(points)
        row = tune(name, points, client, levels)
        rows.append(row)
        curve_str = " ".join(f"L{lv}={r:.0%}" for lv, r in points.items())
        print(f"\n=== {name}  [{row['verdict']}]  {curve_str}")
        print(f"    {row.get('detail', '')}")
        if row.get("error"):
            print(f"    NO PATCH: {row['error']}")
            continue
        print(f"    {row['rationale']}\n    verified: {row['verified']}\n")
        print(row["diff"])
        if a.apply:
            pathlib.Path(row["path"]).write_text(row["patched"])
            print(f"    applied to {row['path']}")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(
            [{k: v for k, v in r.items() if k != "patched"} for r in rows], indent=1))
    good = sum(1 for r in rows if not r.get("error"))
    print(f"\n[tune] {good}/{len(rows)} verified patches"
          + ("" if a.apply else "; re-run with --apply to write them"))


if __name__ == "__main__":
    main()
