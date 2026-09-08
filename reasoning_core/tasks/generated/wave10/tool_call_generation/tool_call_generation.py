from dataclasses import dataclass

import random

from reasoning_core.template import Config, Entry, Task, edict

TASK_META = {'parent_source_id': None,
 'idea': 'tool_call_generation (draw 1 of 2)',
 'hypothesis': 'ASTRA0-07',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave10/tool_call_generation',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1351444575,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


_F = {
    "ship": dict(action="ship a package to", enum=["London", "Paris", "Tokyo", "Miami"],
                 enum_param="destination", bool_param="express",
                 bool_yes="express delivery", bool_no="standard delivery"),
    "order": dict(action="order", enum=["paper", "pens", "notebooks", "desks"],
                  enum_param="product", bool_param="gift",
                  bool_yes="gift wrapping", bool_no="plain packaging"),
    "quote": dict(action="produce a quote for", enum=["steel", "wood", "glass", "copper"],
                  enum_param="material", bool_param="rush",
                  bool_yes="rush turnaround", bool_no="standard turnaround"),
    "import": dict(action="import goods from", enum=["Delhi", "Hanoi", "Lagos", "Lima"],
                   enum_param="origin", bool_param="customs",
                   bool_yes="expedited customs", bool_no="regular customs"),
    "rent": dict(action="rent a", enum=["van", "sedan", "truck", "scooter"],
                 enum_param="vehicle", bool_param="insurance",
                 bool_yes="full insurance", bool_no="basic coverage"),
    "cater": dict(action="cater with", enum=["pizza", "sushi", "bbq", "vegan"],
                  enum_param="menu", bool_param="gratuity",
                  bool_yes="gratuity included", bool_no="gratuity excluded"),
}

_FAMILIES = sorted(_F)


def _pick_families(offered):
    allf = list(_FAMILIES)
    target = random.choice(allf)
    others = [f for f in allf if f != target]
    random.shuffle(others)
    dist = others[:offered - 1]
    chosen = [target] + dist
    return target, sorted(chosen)


def _compute_total(form, unit, qty, base, surch, taxpct):
    if form == "linear":
        return unit * qty + base
    if form == "double":
        return unit * qty + base + surch
    if form == "tax":
        n = (unit * qty + base) * (100 + taxpct)
        return (n + 50) // 100
    raise ValueError(form)


def _call_answer(fam, ev, total, bval):
    s = _F[fam]
    b = "True" if bval else "False"
    return (
        f"{fam}({s['enum_param']}={ev}, total={total}, "
        f"{s['bool_param']}={b})"
    )


def _render_tools(offered):
    lines = []
    for fam in offered:
        s = _F[fam]
        opts = ", ".join(s["enum"])
        lines.append(
            f"{fam}({s['enum_param']}: str in {{{opts}}}, "
            f"total: int > 0, {s['bool_param']}: bool default: False)"
        )
    return "\n".join(lines)


def _render_request(fam, ev, form, unit, qty, base, surch, taxpct, bval):
    s = _F[fam]
    parts = [f"{s['action']} {ev}.",
             f"It is priced at ${unit} for each of {qty} units, "
             f"plus a flat ${base} base fee."]
    if form == "double":
        parts.append(f"A ${surch} service surcharge also applies.")
    if form == "tax":
        parts.append(f"Then a {taxpct}% tax is applied, rounded to the nearest dollar.")
    parts.append(f"Use {s['bool_yes'] if bval else s['bool_no']}.")
    parts.append("Determine the total.")
    return " ".join(parts)


def _canon(v):
    s = str(v).strip()
    low = s.lower()
    if low in ("true", "false"):
        return "true" if low == "true" else "false"
    try:
        return "i:" + str(int(s))
    except ValueError:
        return "s:" + low


def _parse_call(text):
    if not text:
        return None, {}
    s = str(text).strip()
    if "(" not in s or not s.endswith(")"):
        return None, {}
    name, rest = s.split("(", 1)
    name = name.strip()
    if not name:
        return None, {}
    inside = rest[:-1]
    params = {}
    for chunk in inside.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            return None, {}
        k, v = chunk.split("=", 1)
        params[k.strip()] = _canon(v)
    return name, params


@dataclass
class ToolCallGenerationV1Config(Config):
    offered: int = 2
    unit_max: int = 3
    qty_max: int = 3
    base_max: int = 0
    surch_max: int = 0
    allow_double: bool = False
    allow_tax: bool = False

    def apply_difficulty(self, level):
        self.offered = min(5, 2 + level // 2)
        mag = 3 + 4 * level
        self.unit_max = max(2, mag)
        self.qty_max = max(2, mag)
        self.base_max = 6 * level
        self.surch_max = 4 * level
        self.allow_double = level >= 2
        self.allow_tax = level >= 3


class ToolCallGeneration(Task):
    summary = ("Convert a short request into one call of the matching supplied tool, "
               "selecting among distractor tools and filling enum, computed-integer, and "
               "boolean arguments under linear, double, and taxed pricing constraints with "
               "defaults.")
    config_cls = ToolCallGenerationV1Config
    task_version = 2

    def generate_entry(self):
        cfg = self.config
        target, offered = _pick_families(cfg.offered)
        specs = _F[target]
        for _ in range(1000):
            forms = ["linear"]
            if cfg.allow_double:
                forms.append("double")
            if cfg.allow_tax:
                forms.append("tax")
            form = random.choice(forms)
            unit = random.randint(1, cfg.unit_max)
            qty = random.randint(1, cfg.qty_max)
            base = random.randint(0, cfg.base_max)
            surch = random.randint(0, cfg.surch_max)
            taxpct = random.choice([5, 8, 10, 12])
            total = _compute_total(form, unit, qty, base, surch, taxpct)
            if total < 1:
                continue
            ev = random.choice(specs["enum"])
            bval = random.random() < 0.5
            answer = _call_answer(target, ev, total, bval)
            recomputed = _call_answer(
                target, ev, _compute_total(form, unit, qty, base, surch, taxpct), bval)
            if answer != recomputed:
                raise RuntimeError("tool call recompute mismatch")
            payload = {
                "tools": _render_tools(offered),
                "request": _render_request(
                    target, ev, form, unit, qty, base, surch, taxpct, bval),
            }
            metadata = edict({
                "target": target,
                "form": form,
                "total": int(total),
                "payload": payload,
            })
            return Entry(metadata=metadata, answer=answer)
        raise RuntimeError("tool_call_generation: failed to generate an admissible call")

    def render_prompt(self, metadata):
        payload = metadata.payload
        return (
            f"Available tool definitions:\n{payload['tools']}\n\n"
            f"Request: {payload['request']}\n\n"
            f"Call the matching tool with all of its arguments. Write the answer exactly "
            f"as NAME(arg=value, ...) with no spaces around '=', booleans as True/False, "
            f"numbers as integers, and text as its plain value (for example: "
            f"ship(destination=Tokyo, total=132, express=True))."
        )

    def score_answer(self, answer, entry):
        gold_name, gold = _parse_call(entry.answer)
        cand_name, cand = _parse_call(answer)
        if gold_name is None or cand_name is None:
            return 0.0
        if cand_name != gold_name or cand != gold:
            return 0.0
        return 1.0
