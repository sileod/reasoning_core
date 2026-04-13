"""Spatiotemporal reasoning on a 2-D integer grid. z3 verifies answer uniqueness."""

import random
import re
from dataclasses import dataclass
from itertools import combinations

from z3 import Int, Solver, Distinct, And, Abs, Not, sat, unsat

from reasoning_core.template import Task, Problem, Config, edict


@dataclass
class NavigationConfig(Config):
    n_objects: int = 3
    grid: int = 4
    n_steps: int = 1
    n_exact: int = 2
    n_rel: int = 6
    max_tries: int = 80

    def update(self, c=1):
        self.n_objects += 1 * c
        self.grid += 1 * c
        self.n_steps += 1 * c
        self.n_rel += 1 * c


DELTAS = [
    (-2, 0), (-1, 0), (1, 0), (2, 0),
    (0, -2), (0, -1), (0, 1), (0, 2),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
]
HINV = {"left": "right", "right": "left", "aligned": "aligned"}
VINV = {"above": "below", "below": "above", "aligned": "aligned"}
SCORE_RE = {
    "coord": r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?",
    "distance": r"-?\d+",
    "relation": r"(left|right|aligned)\s*,\s*(above|below|aligned)",
}


def hrel(x1, x2):
    return "left" if x1 < x2 else "right" if x1 > x2 else "aligned"


def vrel(y1, y2):
    return "above" if y1 > y2 else "below" if y1 < y2 else "aligned"


def pair_rel(p, q):
    return hrel(p[0], q[0]), vrel(p[1], q[1])


class GridVars:
    """Small symbolic helper over x_t_obj, y_t_obj variables."""

    def __init__(self, names, G, T):
        self.names, self.G, self.T = names, G, T
        self.x = [{a: Int(f"x_{t}_{a}") for a in names} for t in range(T + 1)]
        self.y = [{a: Int(f"y_{t}_{a}") for a in names} for t in range(T + 1)]

    def pos(self, a, t):
        return self.x[t][a], self.y[t][a]

    def coord(self, a, t, p):
        return And(self.x[t][a] == p[0], self.y[t][a] == p[1])

    def dist(self, a, b, t):
        return Abs(self.x[t][a] - self.x[t][b]) + Abs(self.y[t][a] - self.y[t][b])

    def rel(self, a, b, t, axis, r):
        u = self.x[t][a] if axis == "h" else self.y[t][a]
        v = self.x[t][b] if axis == "h" else self.y[t][b]
        if r in ("left", "below"):
            return u < v
        if r in ("right", "above"):
            return u > v
        return u == v

    def code(self, a, t):
        return self.x[t][a] + (self.G + 1) * self.y[t][a]

    def physics(self, solver):
        for t in range(self.T + 1):
            for a in self.names:
                solver.add(0 <= self.x[t][a], self.x[t][a] <= self.G)
                solver.add(0 <= self.y[t][a], self.y[t][a] <= self.G)
            solver.add(Distinct(*[self.code(a, t) for a in self.names]))


def apply_step(pos, st):
    """Apply one step to a map obj -> (x, y); works for ints and z3 exprs."""
    nxt = dict(pos)
    if st["k"] == "move":
        x, y = pos[st["a"]]
        dx, dy = st["d"]
        nxt[st["a"]] = (x + dx, y + dy)
    elif st["k"] == "swap":
        a, b = st["a"], st["b"]
        nxt[a], nxt[b] = pos[b], pos[a]
    else:  # jump
        xb, yb = pos[st["b"]]
        dx, dy = st["d"]
        nxt[st["a"]] = (xb + dx, yb + dy)
    return nxt


def random_step(rng, names):
    k = rng.choice(["move", "swap", "jump"])
    if k == "move":
        return {"k": "move", "a": rng.choice(names), "d": rng.choice(DELTAS)}
    if k == "swap":
        a, b = rng.sample(names, 2)
        return {"k": "swap", "a": a, "b": b}
    a, b = rng.sample(names, 2)
    return {"k": "jump", "a": a, "b": b, "d": rng.choice(DELTAS)}


def step_text(st):
    if st["k"] == "move":
        dx, dy = st["d"]
        return f"{st['a']} moves by ({dx}, {dy})."
    if st["k"] == "swap":
        return f"{st['a']} and {st['b']} swap positions."
    dx, dy = st["d"]
    return f"{st['a']} jumps to {st['b']}'s position offset by ({dx}, {dy})."


def fact_text(f):
    if f["k"] == "coord":
        x, y = f["p"]
        return f"{f['a']} starts at ({x}, {y})."
    if f["k"] == "h":
        return (
            f"{f['a']} is in the same column as {f['b']}."
            if f["r"] == "aligned" else
            f"{f['a']} is {f['r']} of {f['b']}."
        )
    return (
        f"{f['a']} is in the same row as {f['b']}."
        if f["r"] == "aligned" else
        f"{f['a']} is {f['r']} {f['b']}."
    )


def sample_world(rng, names, G, T):
    pts = [(x, y) for x in range(G + 1) for y in range(G + 1)]
    state = dict(zip(names, rng.sample(pts, len(names))))
    states, steps = [state], []

    for _ in range(T):
        for _ in range(60):
            st = random_step(rng, names)
            nxt = apply_step(state, st)
            vals = list(nxt.values())
            if all(0 <= x <= G and 0 <= y <= G for x, y in vals) and len(set(vals)) == len(vals):
                steps.append(st)
                states.append(nxt)
                state = nxt
                break
        else:
            a, b = rng.sample(names, 2)
            st = {"k": "swap", "a": a, "b": b}
            state = apply_step(state, st)
            steps.append(st)
            states.append(state)

    return states, steps


def make_fact_pools(rng, names, state0):
    coords = [{"k": "coord", "a": a, "p": state0[a]} for a in names]
    rels = []

    for a, b in combinations(names, 2):
        h, v = pair_rel(state0[a], state0[b])

        if rng.random() < 0.5:
            rels.append({"k": "h", "a": a, "b": b, "r": h})
        else:
            rels.append({"k": "h", "a": b, "b": a, "r": HINV[h]})

        if rng.random() < 0.5:
            rels.append({"k": "v", "a": a, "b": b, "r": v})
        else:
            rels.append({"k": "v", "a": b, "b": a, "r": VINV[v]})

    return coords, rels


def build_solver(names, facts, steps, G):
    gv = GridVars(names, G, len(steps))
    s = Solver()
    gv.physics(s)

    for f in facts:
        if f["k"] == "coord":
            s.add(gv.coord(f["a"], 0, f["p"]))
        else:
            s.add(gv.rel(f["a"], f["b"], 0, f["k"], f["r"]))

    for t, st in enumerate(steps):
        cur = {a: gv.pos(a, t) for a in names}
        nxt = apply_step(cur, st)
        for a in names:
            s.add(gv.x[t + 1][a] == nxt[a][0], gv.y[t + 1][a] == nxt[a][1])

    return s, gv


def entailed(base_solver, expr):
    s = Solver()
    s.add(base_solver.assertions())
    s.add(Not(expr))
    return s.check() == unsat


def pick_query(rng, solver, gv, names, final):
    T = gv.T
    pairs = list(combinations(names, 2))
    qtypes = ["coord", "distance", "relation"]
    rng.shuffle(qtypes)

    for qt in qtypes:
        if qt == "coord":
            order = names[:]
            rng.shuffle(order)
            for a in order:
                if entailed(solver, gv.coord(a, T, final[a])):
                    x, y = final[a]
                    return {
                        "type": "coord", "a": a,
                        "answer": f"({x}, {y})",
                    }

        elif qt == "distance":
            rng.shuffle(pairs)
            for a, b in pairs:
                d = abs(final[a][0] - final[b][0]) + abs(final[a][1] - final[b][1])
                if entailed(solver, gv.dist(a, b, T) == d):
                    return {
                        "type": "distance", "a": a, "b": b,
                        "answer": str(d),
                    }

        else:
            rng.shuffle(pairs)
            for a, b in pairs:
                h, v = pair_rel(final[a], final[b])
                expr = And(gv.rel(a, b, T, "h", h), gv.rel(a, b, T, "v", v))
                if entailed(solver, expr):
                    return {
                        "type": "relation", "a": a, "b": b,
                        "answer": f"({h}, {v})",
                    }

    return None



class Navigation(Task):

    def __init__(self, config=NavigationConfig()):
        super().__init__(config=config)

    def generate(self):
        rng = random.Random(self.config.seed)
        cfg = self.config

        n = max(2, cfg.n_objects)
        G = cfg.grid
        while (G + 1) ** 2 < n:
            G += 1
        names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:n]

        for attempt in range(cfg.max_tries):
            states, steps = sample_world(rng, names, G, cfg.n_steps)
            state0, final = states[0], states[-1]

            coord_pool, rel_pool = make_fact_pools(rng, names, state0)
            ne = min(rng.randint(0, cfg.n_exact) + attempt // 25, len(coord_pool))
            nr = min(cfg.n_rel + attempt, len(rel_pool))
            facts = rng.sample(coord_pool, ne) + rng.sample(rel_pool, nr)
            rng.shuffle(facts)

            solver, gv = build_solver(names, facts, steps, G)
            if solver.check() != sat:
                continue

            query = pick_query(rng, solver, gv, names, final)
            if query is None:
                continue

            metadata = edict({
                "answer_type": query["type"],
                "query_a": query["a"],
                "query_b": query.get("b"),
                "grid": G,
                "objects": names,
                "facts": facts,
                "steps": steps,
                "initial_state": state0,
                "final_state": final,
            })
            return Problem(metadata=metadata, answer=query["answer"])

        # fallback: reveal all initial coordinates
        states, steps = sample_world(rng, names, G, cfg.n_steps)
        state0, final = states[0], states[-1]
        facts = [{"k": "coord", "a": a, "p": state0[a]} for a in names]
        a = rng.choice(names)
        x, y = final[a]
        metadata = edict({
            "answer_type": "coord",
            "query_a": a,
            "query_b": None,
            "grid": G,
            "objects": names,
            "facts": facts,
            "steps": steps,
            "initial_state": state0,
            "final_state": final,
        })
        return Problem(metadata=metadata, answer=f"({x}, {y})")

    def prompt(self, metadata):
        G = metadata["grid"]
        facts = metadata["facts"]
        steps = metadata["steps"]
        a = metadata["query_a"]
        b = metadata.get("query_b")
        kind = metadata["answer_type"]

        facts_txt = "\n".join(f"- {fact_text(f)}" for f in facts)
        steps_txt = "\n".join(f"{i+1}. {step_text(st)}" for i, st in enumerate(steps)) if steps else "None."

        if kind == "coord":
            question = f"What is the final coordinate of {a}? Answer as (x, y)."
        elif kind == "distance":
            question = f"What is the final Manhattan distance between {a} and {b}? Answer as an integer."
        else:
            question = (
                f"What is the final spatial relation of {a} to {b}? "
                f"The answer is (horizontal, vertical), where horizontal is "
                f"left/right/aligned and vertical is above/below/aligned."
            )

        return (
            f"Objects occupy distinct points on the integer grid [0, {G}] x [0, {G}].\n"
            f"North is +y and East is +x. Any object not mentioned in a step stays fixed.\n\n"
            f"Initial facts:\n{facts_txt}\n\n"
            f"Steps:\n{steps_txt}\n\n"
            f"{question}\n"
        )

    def score_answer(self, answer, entry):
        kind = entry.metadata["answer_type"]
        pa = re.search(SCORE_RE[kind], str(answer).lower())
        pg = re.search(SCORE_RE[kind], entry.answer.lower())
        if not pa or not pg:
            return 0.0
        return float(pa.group() == pg.group() if kind == "distance" else pa.groups() == pg.groups())