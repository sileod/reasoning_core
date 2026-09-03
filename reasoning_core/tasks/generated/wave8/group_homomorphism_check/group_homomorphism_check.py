from dataclasses import dataclass

import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


@dataclass
class GroupHomomorphismCheckConfig(Config):
    n: int = 6
    m: int = 6

    def apply_difficulty(self, level):
        self.n = sround(self.n + 3 * level)
        self.m = sround(self.m + 3 * level)


def _cyclic_table(order):
    return [[(i + j) % order for j in range(order)] for i in range(order)]


def _render_table(table):
    return "\n".join(" ".join(str(v) for v in row) for row in table)


def _render_mapping(f):
    return " ".join(str(v) for v in f)


def _violation_count(f, domain_table, codomain_table):
    n = len(domain_table)
    count = 0
    for x in range(n):
        fx = f[x]
        for y in range(n):
            if f[domain_table[x][y]] != codomain_table[fx][f[y]]:
                count += 1
    return count


def _parse_int(answer):
    return int(str(answer).strip())


class GroupHomomorphismCheck(Task):
    summary = ("Count the ordered pairs of a mapped finite cyclic group for which the homomorphism "
               "condition fails (0 iff the mapping is a homomorphism), over genuine homomorphisms "
               "and perturbed candidate maps.")
    config_cls = GroupHomomorphismCheckConfig

    def generate_entry(self):
        n = self.config.n
        m = self.config.m
        domain_table = _cyclic_table(n)
        codomain_table = _cyclic_table(m)

        d = random.randrange(m)
        f = [(d * x) % m for x in range(n)]
        k = random.randrange(n)
        for p in random.sample(range(n), k):
            f[p] = random.randrange(m)

        count = _violation_count(f, domain_table, codomain_table)

        payload = {
            "n": n,
            "m": m,
            "domain_table": domain_table,
            "codomain_table": codomain_table,
            "mapping": f,
        }
        metadata = edict({"payload": payload, "count": count})
        return Entry(metadata=metadata, answer=str(count))

    def render_prompt(self, metadata):
        payload = metadata.payload
        return (
            f"The domain is the cyclic group of order {payload['n']} under addition mod {payload['n']}; "
            f"its Cayley table is:\n{_render_table(payload['domain_table'])}\n"
            f"The codomain is the cyclic group of order {payload['m']} under addition mod {payload['m']}; "
            f"its Cayley table is:\n{_render_table(payload['codomain_table'])}\n"
            f"The candidate map f sends each element of the domain to an element of the codomain, "
            f"listed as f(0) f(1) ... f(n-1):\nf = [{_render_mapping(payload['mapping'])}]\n"
            f"For every ordered pair (x, y) of domain elements the map is a homomorphism exactly when "
            f"f(x\u00b7y) = f(x)\u00b7f(y). Count the number of ordered pairs (x, y) for which this "
            f"condition fails (this count is 0 if and only if f is a homomorphism).\n"
            f"The answer is the integer count."
        )

    def score_answer(self, answer, entry):
        try:
            return 1.0 if _parse_int(answer) == _parse_int(entry.answer) else 0.0
        except (ValueError, TypeError):
            return 0.0

    def distractor_candidates(self, entry):
        c = _parse_int(entry.answer)
        n = len(entry.metadata.payload["domain_table"])
        total = n * n
        seen = set()
        out = []
        for val in (c + 1, c - 1, total - c, c + 2, c - 2, n):
            if 0 <= val <= total and val not in seen:
                seen.add(val)
                out.append(str(val))
        return out


TASK_META = {'parent_source_id': None,
 'idea': 'group_homomorphism_check (draw 1 of 2)',
 'hypothesis': 'W1-023',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/group_homomorphism_check',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1330648825,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
