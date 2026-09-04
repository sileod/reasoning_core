import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict

TR, FA, UN = "True", "False", "Unknown"
AND, OR, NOT = "and", "or", "not"


def _not(v):
    return {TR: FA, FA: TR, UN: UN}[v]


def _and(a, b):
    if a == FA or b == FA:
        return FA
    if a == TR and b == TR:
        return TR
    return UN


def _or(a, b):
    if a == TR or b == TR:
        return TR
    if a == FA and b == FA:
        return FA
    return UN


def _apply(op, a, b=None):
    if op == NOT:
        return _not(a)
    if op == AND:
        return _and(a, b)
    return _or(a, b)


def _render_gate(op, a, b):
    if op == "input":
        return a
    if op == NOT:
        return "not " + a
    return f"{a} {op} {b}"


@dataclass
class ThreeValuedLogicPropagationConfig(Config):
    n_inputs: int = 3
    n_gates: int = 4
    depth: int = 3

    def apply_difficulty(self, level):
        self.n_inputs = 3 + level
        self.n_gates = 4 + 2 * level
        self.depth = 3 + level


class ThreeValuedLogicPropagation(Task):
    summary = ("Evaluate three-valued-logic (True/False/Unknown) gate networks and hybrid "
               "gate-plus-expression circuits and report the propagated value of every gate "
               "after nested operations.")
    config_cls = ThreeValuedLogicPropagationConfig

    def generate_entry(self):
        mode = random.choice(["net", "hybrid"])
        n = int(self.config.n_inputs)
        ng = int(self.config.n_gates)
        names = [f"in{i}" for i in range(n)] + [f"g{i}" for i in range(ng)]
        gates = {}
        for i in range(n):
            gates[f"in{i}"] = ("input", random.choice([TR, FA, UN]), None)
        for i in range(ng):
            op = random.choice([AND, OR, NOT])
            a = random.choice(names[: n + i])
            if op == NOT:
                gates[f"g{i}"] = (NOT, a, None)
            else:
                b = random.choice(names[: n + i])
                gates[f"g{i}"] = (op, a, b)

        vals = _eval_net(gates, names)

        lines = [f"{name} = {_render_gate(gates[name][0], gates[name][1], gates[name][2])}"
                 for name in names]
        gate_order = [name for name in names if name.startswith("g")]
        expr_text = None

        if mode == "hybrid":
            depth = int(self.config.depth)
            expr = _random_expr(random, depth)
            expr_text = _render_expr(expr)
            expr_val = _eval_expr(expr)
            cur = vals[f"g{ng - 1}"]
            for r in range(depth):
                gname = f"g{ng + r}"
                lines.append(f"{gname} = g{ng - 1} and (EXPR)")
                cur = _and(cur, expr_val)
                vals[gname] = cur
                gate_order.append(gname)

        answer = "; ".join(vals[gn] for gn in gate_order)

        metadata = edict({"mode": mode})
        metadata.payload = {"circuit": "\n".join(lines),
                            "expr": expr_text if expr_text is not None else ""}
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        rules = ("Each of True, False, and Unknown is a valid truth value. "
                 "not True = False, not False = True, not Unknown = Unknown; "
                 "and returns False if either operand is False, True only if both are True, "
                 "otherwise Unknown; "
                 "or returns True if either operand is True, False only if both are False, "
                 "otherwise Unknown.")
        header = (f"Under three-valued logic with True, False, and Unknown, a network of gates "
                  f"holds a truth value per named input and gate. Evaluate bottom-up using the "
                  f"rules below.\n\nRules: {rules}\n\n")
        circ = metadata.payload["circuit"]
        if metadata.mode == "hybrid":
            expr = metadata.payload["expr"]
            circ = circ.replace("(EXPR)", f"({expr})")
        return (header + f"{circ}\n\n"
                f"Report the propagated value of every gate g0, g1, ... in order, as a "
                f"semicolon-separated list e.g. True; Unknown; False. "
                f"The answer is exactly that list.")

    def score_answer(self, answer, entry):
        gold = entry.answer
        if isinstance(answer, str):
            answer = " ".join(answer.strip().strip(".").split())
            gold = " ".join(gold.split())
        return 1.0 if answer == gold else 0.0


def _eval_expr(expr):
    if isinstance(expr, str):
        return expr
    if expr[0] == NOT:
        return _not(_eval_expr(expr[1]))
    op, left, right = expr
    return _apply(op, _eval_expr(left), _eval_expr(right))


def _render_expr(expr):
    if isinstance(expr, str):
        return expr
    if expr[0] == NOT:
        return "not (" + _render_expr(expr[1]) + ")"
    op, left, right = expr
    return "(" + _render_expr(left) + " " + op + " " + _render_expr(right) + ")"


def _random_expr(rng, depth):
    if rng.random() < 0.5 or depth <= 1:
        return rng.choice([TR, FA, UN])
    op = rng.choice([NOT, AND, OR])
    if op == NOT:
        return (NOT, _random_expr(rng, depth - 1))
    return (op, _random_expr(rng, depth - 1), _random_expr(rng, depth - 1))


def _eval_net(gates, names):
    vals = {}
    pending = list(names)
    while pending:
        progressed = False
        still = []
        for name in pending:
            op, a, b = gates[name]
            if op == "input":
                vals[name] = a
                progressed = True
            elif a in vals and (op == NOT or b in vals):
                vals[name] = _apply(op, vals[a], vals.get(b))
                progressed = True
            else:
                still.append(name)
        if not progressed:
            raise RuntimeError("circuit cycle - topological order violated")
        pending = still
    return vals


TASK_META = {'parent_source_id': None,
 'idea': 'three_valued_logic_propagation (draw 1 of 1)',
 'hypothesis': 'HV-055',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/three_valued_logic_propagation',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1851018107,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
