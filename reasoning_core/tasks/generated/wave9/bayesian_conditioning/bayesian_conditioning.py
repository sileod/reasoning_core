from dataclasses import dataclass
from fractions import Fraction
import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload

TASK_META = {'parent_source_id': None,
 'idea': 'bayesian_conditioning (draw 1 of 1)',
 'hypothesis': 'HV-001',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/bayesian_conditioning',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1662004003,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def score_fraction(answer, entry):
    """Exact-match scorer for a reduced fraction a/b."""
    ref = entry.answer if hasattr(entry, "answer") else entry["answer"]
    try:
        ref_f = Fraction(ref)
    except Exception:
        return 0.0
    text = str(answer).strip()
    pred = None
    if "/" in text:
        try:
            pred = Fraction(text)
        except Exception:
            return 0.0
    else:
        try:
            pred = Fraction(float(text))
        except Exception:
            return 0.0
    return float(pred == ref_f)


@dataclass
class BayesianConfig(Config):
    level: int = 0
    max_den: int = 9
    n_nodes: int = 4

    def apply_difficulty(self, level):
        self.level = level
        self.max_den = int(5 + 4 * level)
        self.n_nodes = min(3 + level, 6)


class BayesianConditioning(Task):
    summary = ("Compute exact posterior probabilities in small discrete Bayesian "
               "networks from CPTs and observed evidence, returning one queried "
               "marginal as a reduced fraction.")

    config_cls = BayesianConfig

    def generate_entry(self):
        cfg = self.config
        max_den = cfg.max_den
        n_nodes = cfg.n_nodes
        names = ["A", "B", "C", "D", "E", "F", "G", "H"]
        nodes = names[:n_nodes]

        while True:
            parents = {nd: [] for nd in nodes}
            for i, nd in enumerate(nodes):
                if i == 0:
                    continue
                n_par = random.randint(1, min(2, i))
                parents[nd] = random.sample(nodes[:i], n_par)

            all_nodes = list(nodes)
            random.shuffle(all_nodes)
            if len(all_nodes) < 3:
                continue
            ev_node = all_nodes[0]
            query_node = all_nodes[1]
            ev_val = random.randint(0, 1)

            cpts = {}
            for nd in nodes:
                par = parents[nd]
                table = {}
                for mask in range(1 << len(par)):
                    p1 = random.randint(1, max_den - 1)
                    table[mask] = (max_den - p1, p1)
                cpts[nd] = table

            joint = {}
            for s in range(1 << n_nodes):
                assign = {}
                for k, nd in enumerate(nodes):
                    assign[nd] = (s >> k) & 1
                joint[s] = assign

            ev_states = {s: a for s, a in joint.items()
                         if a[ev_node] == ev_val}
            tot = Fraction(0, 1)
            pos = Fraction(0, 1)
            for s, a in ev_states.items():
                pr = Fraction(1, 1)
                for nd in nodes:
                    par = parents[nd]
                    mask = 0
                    for k, p in enumerate(par):
                        mask |= (a[p] << k)
                    p0, p1 = cpts[nd][mask]
                    pr *= Fraction(p0 if a[nd] == 0 else p1, max_den)
                tot += pr
                if a[query_node] == 1:
                    pos += pr

            if tot == 0:
                continue
            res = Fraction(pos, tot)
            if res.denominator <= 1 or res.numerator <= 0 or res.numerator >= res.denominator:
                continue

            p0, p1 = res.numerator, res.denominator
            zero_line = f"P({query_node}=0 | evidence)={p0}/{p1}"
            break

        # Render CPT lines (parents in topological order given by index)
        lines = []
        for nd in nodes:
            par = parents[nd]
            pars_str = "".join(par) if par else "-"
            lines.append(f"{nd} (parents {pars_str}):")
            for mask in range(1 << len(par)):
                cp0, cp1 = cpts[nd][mask]
                if par:
                    vals = ", ".join(f"{p}=" + str((mask >> parents[nd].index(p)) & 1)
                                     for p in par)
                    given_str = f"{vals}"
                else:
                    given_str = "-"
                lines.append(f"  P({nd}=0|{given_str})={cp0}/{max_den}, "
                             f"P({nd}=1|{given_str})={cp1}/{max_den}")

        ev_str = f"{ev_node}={ev_val}"
        prompt = (
            "A small discrete Bayesian network has binary nodes "
            f"{', '.join(nodes)}. The conditional probability tables are:\n"
            + "\n".join(lines)
            + f"\n\nWe observe the evidence {ev_str}.\n"
            f"What is the posterior probability P({query_node}=1 | {ev_str})?\n"
            "Give the answer as a reduced fraction of the form a/b (e.g. 3/7)."
        )

        payload = {
            "nodes": nodes,
            "parents": {nd: list(par) for nd, par in parents.items()},
            "cpt_lines": lines,
            "evidence": ev_str,
            "query": query_node,
        }
        metadata = edict({
            "nodes": nodes,
            "parents": {nd: list(par) for nd, par in parents.items()},
            "evidence": {ev_node: ev_val},
            "query": query_node,
            "answer_str": f"{p0}/{p1}",
            "payload": payload,
        })
        metadata.payload = payload
        answer = f"{p0}/{p1}"

        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        payload = metadata.payload
        topo = ", ".join(payload["nodes"])
        return ("A small discrete Bayesian network has binary nodes "
                f"{topo}. The conditional probability tables are:\n"
                + "\n".join(payload["cpt_lines"])
                + f"\n\nWe observe the evidence {payload['evidence']}.\n"
                f"What is the posterior probability P({payload['query']}=1 | "
                f"{payload['evidence']})?\n"
                "Give the answer as a reduced fraction of the form a/b (e.g. 3/7).")

    def score_answer(self, answer, entry):
        return score_fraction(answer, entry)
