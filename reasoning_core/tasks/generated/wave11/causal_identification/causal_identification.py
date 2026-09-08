"""Causal identification: which candidate mechanisms remain possible after interventions/observations.

We model a set of binary variables with a set of candidate causal mechanisms.  Each
candidate mechanism is a directed acyclic graph (a DAG) over the same named binary
variables.  The truth is one of these candidates.  Given an ordered sequence of root
interventions and the full observed outcomes under each, identify which candidates remain
possible (i.e. could have produced every observed outcome).

Variable dynamics: each mechanism is a linear sum-mod-2 causal model.  The value of a
node is the XOR of the values of its parents (using the intervened value for the
intervened node) XOR a per-node constant bias; a root node (no parents) takes its bias
as its value.
"""

from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload

TASK_META = {'parent_source_id': None,
 'idea': 'causal_identification (draw 1 of 2)',
 'hypothesis': 'ASTRA2-causal_identification',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave11/causal_identification',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3090820539,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _topo_sort(nodes, edges):
    import heapq
    children = {n: set() for n in nodes}
    indeg = {n: 0 for n in nodes}
    for (a, b) in edges:
        children[a].add(b)
        indeg[b] += 1
    ready = [n for n in nodes if indeg[n] == 0]
    heapq.heapify(ready)
    indeg_c = dict(indeg)
    order = []
    while ready:
        n = heapq.heappop(ready)
        order.append(n)
        for c in sorted(children[n]):
            indeg_c[c] -= 1
            if indeg_c[c] == 0:
                heapq.heappush(ready, c)
    return order


def _simulate(edges, bias, inter, ordered_nodes):
    val = {}
    parents = {}
    for (a, b) in edges:
        parents.setdefault(b, []).append(a)
    for n in ordered_nodes:
        if n in inter:
            val[n] = inter[n]
            continue
        x = bias.get(n, 0)
        for p in parents.get(n, []):
            x ^= val[p]
        val[n] = x
    return val


def _render_mechanisms(candidates, labels):
    lines = []
    for ci, (edges, bias) in enumerate(candidates):
        pm = {}
        for (a, b) in edges:
            pm.setdefault(b, []).append(a)
        nodes = sorted(set([e[0] for e in edges]) | set([e[1] for e in edges]))
        for v in bias:
            nodes.append(v)
        nodes = sorted(set(nodes))
        desc = []
        for v in nodes:
            pars = pm.get(v, [])
            if pars:
                desc.append(f"{v} = XOR({', '.join(sorted(pars))})")
            else:
                desc.append(f"{v} = {bias.get(v, 0)}")
        lines.append(f"{labels[ci]}: " + "; ".join(desc))
    return lines


@dataclass
class CausalIdentificationConfig(Config):
    n_candidates: int = 3
    n_vars: int = 4
    n_ops: int = 2
    depth: int = 2

    def apply_difficulty(self, level):
        self.n_candidates = 3 + int(level // 2)
        self.n_vars = 4 + (level + 1) // 2
        self.n_ops = 2 + (level * 4) // 5
        self.depth = 1 + int(level * 0.4)


def _parse_answer(answer):
    if answer is None:
        return None
    s = str(answer).strip()
    if s == "":
        return None
    if s == "none":
        return ("none",)
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return None
    return tuple(sorted(parts))


class CausalIdentification(Task):
    summary = ("Identify which candidate linear-mod-2 causal DAGs remain possible after an "
               "ordered sequence of root interventions and full observations, restoring the "
               "sorted candidate-name list whose XOR/bias outcomes match every step across "
               "varied DAG topologies, candidate counts, and intervention depths.")
    config_cls = CausalIdentificationConfig
    task_version = 2

    def generate_entry(self):
        import random
        cfg = self.config
        n_c = int(cfg.n_candidates)
        n_ops = int(cfg.n_ops)
        depth = int(cfg.depth)
        pool = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]
        n_v = int(cfg.n_vars)
        if n_v > len(pool):
            n_v = len(pool)
        var_names = sorted(random.sample(pool, n_v))

        candidates = []
        for _ci in range(n_c):
            perm = list(var_names)
            random.shuffle(perm)
            edges = []
            for i in range(n_v):
                node = perm[i]
                max_par = min(depth, i)
                if max_par < 1:
                    continue
                n_par = random.randint(1, max_par)
                for p in random.sample(perm[:i], n_par):
                    edges.append((p, node))
            bias = {v: random.randint(0, 1) for v in var_names}
            candidates.append((tuple(sorted(edges)), bias))

        true_idx = random.randrange(n_c)
        truth_edges, truth_bias = candidates[true_idx]
        ordered_nodes = _topo_sort(var_names, truth_edges)

        roots = [v for v in var_names if not any(b == v for (a, b) in truth_edges)]
        if not roots:
            roots = list(var_names)
        inter_pool = [(r, v) for r in roots for v in (0, 1)]
        random.shuffle(inter_pool)
        chosen = inter_pool[:n_ops]
        if len(chosen) < 2:
            chosen = inter_pool[:1] * 1
        ops = []
        outcomes = []
        for (iv, iv_val) in chosen:
            val = _simulate(truth_edges, truth_bias, {iv: iv_val}, ordered_nodes)
            outcome = " ".join(f"{v}={val[v]}" for v in var_names)
            ops.append((iv, iv_val))
            outcomes.append(outcome)

        possible = []
        for ci, (cedges, cbias) in enumerate(candidates):
            c_order = _topo_sort(var_names, cedges)
            ok = True
            for (iv, iv_val), outcome in zip(ops, outcomes):
                val = _simulate(cedges, cbias, {iv: iv_val}, c_order)
                out_str = " ".join(f"{v}={val[v]}" for v in var_names)
                if out_str != outcome:
                    ok = False
                    break
            if ok:
                possible.append(ci)

        cand_labels = [f"C{ci}" for ci in range(n_c)]
        ans = sorted(cand_labels[ci] for ci in possible)
        answer = ",".join(ans) if ans else "none"

        meta = edict({
            "variables": var_names,
            "candidates": n_c,
            "true_index": true_idx,
            "ops": [(iv, v) for (iv, v) in ops],
            "outcomes": outcomes,
        })
        meta.payload = {
            "variables": var_names,
            "candidates": cand_labels,
            "mechanisms": _render_mechanisms(candidates, cand_labels),
            "ops": [(f"intervene {iv} = {v}", out) for (iv, v), out in zip(ops, outcomes)],
        }
        return Entry(metadata=meta, answer=answer)

    def render_prompt(self, metadata):
        lines = []
        lines.append("We have a set of binary variables " + ", ".join(metadata.variables) + ".")
        lines.append("Each candidate mechanism below defines how each variable's value is "
                     "determined (XOR of its parents, or a fixed constant for a root).")
        lines.append("One of the candidate mechanisms is the true one, but we do not know which.")
        lines.append("")
        lines.append("Candidate mechanisms:")
        for m in metadata.payload["mechanisms"]:
            lines.append("  " + m)
        lines.append("")
        lines.append("We performed the following ordered interventions, observing all variables "
                     "after each:")
        for op, out in metadata.payload["ops"]:
            lines.append(f"  {op} -> observed: {out}")
        lines.append("")
        lines.append("Which candidate mechanism(s) remain possible (consistent with every "
                     "observation)? Give the answer as the sorted, comma-separated list of "
                     "candidate names (e.g. \"C0,C2\"), or the single word \"none\" if none "
                     "remain possible.")
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        gold = _parse_answer(entry.answer)
        got = _parse_answer(answer)
        if got is None:
            return 0.0
        return 1.0 if got == gold else 0.0
