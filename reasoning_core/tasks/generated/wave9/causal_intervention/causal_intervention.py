"""Execute do-interventions in finite acyclic structural causal models.

Given a small parametric structural causal model (SCM) with a fixed topological
order, a query that asks for the value of a target node after we set a subset of
nodes to constants (a do-intervention), the model returns the resulting target
value or the interventional probability of a binary event.

The model is built from deterministic linear structural equations with Gaussian
noise. Because the graph is acyclic and we truncate the model at the intervened
nodes (the do-calculus rule for deterministic SCMs), every value is a computable
deterministic function of the noise terms and the fixed interventions.  For a
binary event we integrate out the Gaussian noise analytically via the standard
normal CDF.

We forward-propagate the truncated structural equations and compute Gaussian
moments exactly, and independently check the result with a Monte-Carlo free
propagation of the interventional model.
"""

import math
import random

from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, edict


@dataclass
class CausalInterventionConfig(Config):
    n_nodes: int = 4
    n_parents_max: int = 2
    n_intervened: int = 1
    n_query_value: int = 2
    n_query_prob: int = 1
    noise_scale: float = 1.0

    def apply_difficulty(self, level):
        self.n_nodes = 4 + level
        self.n_parents_max = 2 + (level >= 3)
        self.n_intervened = 1 + (level >= 2) + (level >= 5)
        self.n_query_value = 2 + (level >= 4)
        self.n_query_prob = 1


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


TASK_META = {'parent_source_id': None,
 'idea': 'causal_intervention (draw 1 of 1)',
 'hypothesis': 'HV-008',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/causal_intervention',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3320301215,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


class CausalIntervention(Task):
    summary = ("Execute do-interventions in finite acyclic structural causal models: "
               "set a subset of nodes to constants, truncate incoming edges, and return "
               "the resulting deterministic target value or the interventional Gaussian "
               "probability of a binary event, obeying the topological truncation rule.")
    config_cls = CausalInterventionConfig

    def generate_entry(self):
        cfg = self.config
        n = cfg.n_nodes
        # Topological order 0..n-1; parents have smaller indices.
        parents = [[] for _ in range(n)]
        intercept = [0.0] * n
        for i in range(n):
            pool = list(range(i))
            n_parents = random.randint(0, min(cfg.n_parents_max, len(pool)))
            chosen = random.sample(pool, n_parents)
            for p in sorted(chosen):
                coef = random.choice([0.5, 1.0, -0.5, -1.0, 0.3, 2.0])
                parents[i].append((p, coef))
            intercept[i] = random.choice([-2.0, -1.0, 0.0, 1.0, 2.0])

        # do-intervention fixes a subset of nodes to constants.
        intervened = set(random.sample(range(n), min(cfg.n_intervened, n)))
        fixed = {}
        for node in intervened:
            fixed[node] = random.choice([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

        # Full Gaussian solve under the truncated (interventional) model.
        # A node's parents can be correlated (shared ancestors), so we carry a
        # full mean vector and covariance matrix, not just marginal variances.
        def truncated_solve(fixed_vals):
            mu = [0.0] * n
            # Sigma as list-of-lists; Sigma[i][j] valid for j <= i.
            sigma = [[0.0] * n for _ in range(n)]
            for i in range(n):
                if i in fixed_vals:
                    mu[i] = fixed_vals[i]
                else:
                    m = intercept[i]
                    for (p, b) in parents[i]:
                        m += b * (fixed_vals[p] if p in fixed_vals else mu[p])
                    mu[i] = m
                    # cov(X_i, X_m) for m < i.
                    for m in range(i):
                        s = 0.0
                        for (p, b) in parents[i]:
                            s += b * sigma[p][m]
                        sigma[i][m] = s
                        sigma[m][i] = s
                    # var(X_i).
                    v = cfg.noise_scale ** 2
                    for (p, b) in parents[i]:
                        for (l, bl) in parents[i]:
                            lo, hi = (l, p) if l < p else (p, l)
                            v += b * bl * sigma[hi][lo]
                    sigma[i][i] = v
            return mu, sigma

        mu, sigma = truncated_solve(fixed)

        # Queries.
        value_nodes = [i for i in range(n) if i not in intervened]
        queries = []
        nq = min(cfg.n_query_value, len(value_nodes))
        for q in random.sample(value_nodes, nq):
            queries.append(("value", q))
        nqp = min(cfg.n_query_prob, len(value_nodes))
        for q in random.sample(value_nodes, nqp):
            threshold = random.choice([0.0, 1.0, 2.0])
            queries.append(("prob", q, threshold))

        answers = []
        p_raw = []
        for query in queries:
            if query[0] == "value":
                _, node = query
                answers.append(round(mu[node], 4))
                p_raw.append(None)
            else:
                _, node, thr = query
                var = sigma[node][node]
                if var <= 0:
                    p = 1.0 if mu[node] > thr else 0.0
                else:
                    p = 1.0 - _normal_cdf((thr - mu[node]) / math.sqrt(var))
                answers.append(round(p, 4))
                p_raw.append(p)
                assert 0.0 <= answers[-1] <= 1.0, "probability out of domain"

        # Independent Monte-Carlo verification, tolerance scaled by analytics.
        # A statistical misfire regenerates the whole instance (bounded retries).
        n_mc = 8000
        for _attempt in range(20):
            ok = True
            for qi, query in enumerate(queries):
                if query[0] == "value":
                    node = query[1]
                    tol = 5.0 * math.sqrt(max(sigma[node][node], 1e-6) / n_mc) + 0.01
                else:
                    node, thr = query[1], query[2]
                    p = p_raw[qi]
                    tol = 5.0 * math.sqrt(max(p * (1.0 - p), 1e-6) / n_mc) + 0.01
                acc = 0.0
                cnt = 0
                for _ in range(n_mc):
                    vals = {}
                    for i in range(n):
                        if i in fixed:
                            vals[i] = fixed[i]
                        else:
                            v = intercept[i] + random.gauss(0.0, cfg.noise_scale)
                            for (p, b) in parents[i]:
                                v += b * (fixed[p] if p in fixed else vals[p])
                            vals[i] = v
                    if query[0] == "value":
                        acc += vals[node]
                    elif vals[node] > thr:
                        cnt += 1
                if query[0] == "value":
                    mc_mean = acc / n_mc
                    if abs(mc_mean - answers[qi]) >= tol:
                        ok = False
                        break
                else:
                    mc_p = cnt / n_mc
                    if abs(mc_p - answers[qi]) >= tol:
                        ok = False
                        break
            if ok:
                break
        else:
            raise RuntimeError("MC verification could not confirm the answer")

        # Build JSON-serializable metadata (string keys everywhere).
        graph = {str(i): [{"parent": p, "coef": b} for (p, b) in parents[i]]
                 for i in range(n)}
        intercepted = {str(i): intercept[i] for i in range(n)}
        intervened_map = {str(k): v for k, v in fixed.items()}
        query_repr = []
        for query in queries:
            if query[0] == "value":
                query_repr.append({"type": "value", "node": query[1]})
            else:
                query_repr.append({"type": "prob", "node": query[1], "threshold": query[2]})

        payload = {
            "nodes": sorted(graph.keys()),
            "graph": graph,
            "intercepts": intercepted,
            "intervened": intervened_map,
            "noise_scale": cfg.noise_scale,
            "queries": query_repr,
        }
        metadata = edict({"payload": payload, "gold_answers": answers})

        answer = " ".join(f"{a}" for a in answers)
        metadata.answer = answer
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        payload = metadata.payload
        graph = payload["graph"]
        intercepted = payload["intercepts"]
        intervened = payload["intervened"]
        queries = payload["queries"]
        noise_scale = payload["noise_scale"]

        nodes = sorted(graph.keys(), key=int)
        lines = []
        lines.append("Consider a structural causal model over nodes "
                     + ", ".join(f"X_{k}" for k in nodes)
                     + ", ordered topologically (a parent always has a smaller index than its child).")
        lines.append("Each node X_i follows the structural equation")
        lines.append("X_i = c_i + sum_{j in pa(i)} beta_ij * X_j + eps_i,  "
                     "eps_i ~ N(0, " + str(noise_scale) + ") independent.")
        lines.append("The structural equations are:")
        for i in nodes:
            terms = [f"{x['coef']}*X_{x['parent']}" for x in sorted(graph[i], key=lambda e: e["parent"])]
            c = intercepted[i]
            if terms:
                lines.append("X_" + str(i) + " = " + str(c) + " + " + " + ".join(terms) + " + eps_" + str(i))
            else:
                lines.append("X_" + str(i) + " = " + str(c) + " + eps_" + str(i))
        lines.append("A do-intervention sets some nodes to constants and deletes their incoming edges:")
        for k, v in sorted(intervened.items(), key=lambda kv: int(kv[0])):
            lines.append("do(X_" + str(int(k)) + " = " + f"{v}" + ")")
        lines.append("")
        for idx, q in enumerate(queries):
            if q["type"] == "value":
                lines.append(f"Query A{idx+1}: compute the value of X_{q['node']} after the intervention.")
            else:
                lines.append(f"Query B{idx+1}: compute the probability that X_{q['node']} > {q['threshold']} "
                             "after the intervention (a number between 0 and 1).")
        lines.append("")
        if len(queries) == 1:
            lines.append("The answer is a single number.")
        else:
            lines.append("The answer is the list of values in query order, separated by spaces.")
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        golds = entry.metadata.gold_answers
        try:
            if len(golds) == 1:
                return 1.0 if round(float(answer.strip()), 4) == golds[0] else 0.0
            parts = answer.strip().split()
            if len(parts) != len(golds):
                return 0.0
            for part, gold in zip(parts, golds):
                if round(float(part), 4) != gold:
                    return 0.0
            return 1.0
        except (ValueError, AttributeError):
            return 0.0
