from dataclasses import dataclass
from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'union_find_representative (draw 1 of 2)',
 'hypothesis': 'W1-014',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/union_find_representative',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3837604770,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class UnionFindConfig(Config):
    n_nodes: int = 6
    n_ops: int = 8
    min_comp: int = 2
    seed: int = 0

    def apply_difficulty(self, level):
        self.n_nodes = sround(self.n_nodes + 3 * level)
        self.n_ops = sround(self.n_ops + 3 * level)
        self.min_comp = sround(self.min_comp + level // 2)


def run_ops(ops, n, tie_smaller):
    parent = list(range(n))
    rank = [0] * n
    last_q = None
    for kind, a, b in ops:
        if kind == 'u':
            ra, rb = find_plain(parent, a), find_plain(parent, b)
            if ra == rb:
                continue
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                choose_a = ra < rb if tie_smaller else ra > rb
                if choose_a:
                    parent[rb] = ra
                    rank[ra] += 1
                else:
                    parent[ra] = rb
                    rank[rb] += 1
        else:
            last_q = a
    return parent, rank, last_q


def find_plain(parent, i):
    while parent[i] != i:
        i = parent[i]
    return i


def q_rep(parent, rank, elem):
    return find_plain(parent, elem)


class UnionFindRepresentative(Task):
    summary = ("Execute canonical union-by-rank on integer-labeled sets from a random sequence of "
               "union and query operations and output the root/representative of a queried element; "
               "same-rank ties are broken toward the smaller or the larger root label, randomized per "
               "instance and stated in the prompt.")
    config_cls = UnionFindConfig

    def generate_entry(self):
        import random
        n = self.config.n_nodes
        n_ops = self.config.n_ops
        nodes = list(range(n))
        tie_smaller = random.random() < 0.5

        for _attempt in range(200):
            ops = []
            n_union = 0
            for _ in range(n_ops):
                if random.random() < 0.6 and n_union < n - 1:
                    a = random.choice(nodes)
                    b = random.choice(nodes)
                    while a == b:
                        b = random.choice(nodes)
                    ops.append(('u', a, b))
                    n_union += 1
                else:
                    ops.append(('q', random.choice(nodes), None))

            parent, rank, last_q = run_ops(ops, n, tie_smaller)
            if last_q is None:
                continue
            n_distinct = len({find_plain(parent, i) for i in range(n)})
            if n_distinct < self.config.min_comp:
                continue
            answer = q_rep(parent, rank, last_q)
            if answer == last_q:
                continue
            if answer == n - 1:
                continue
            if answer == 0:
                continue
            metadata = edict({
                "n": n,
                "ops": [(k, a, b) for (k, a, b) in ops],
                "query": last_q,
                "tie_smaller": tie_smaller,
            })
            metadata.payload = {"n": n, "query": last_q, "ops": metadata.ops, "tie_smaller": tie_smaller}
            return Entry(metadata=metadata, answer=str(answer))

        raise RuntimeError("could not find valid instance")

    def render_prompt(self, metadata):
        tie = ("smaller-labeled root" if metadata.tie_smaller
               else "larger-labeled root")
        lines = []
        lines.append(f"We have a set of {metadata.n} elements labeled 0 through {metadata.n - 1}, "
                     "initially each in its own set.")
        lines.append("Operations are applied left to right:")
        for kind, a, b in metadata.ops:
            if kind == 'u':
                lines.append(f"  union {a} with {b}")
            else:
                lines.append(f"  query {a}")
        lines.append("")
        lines.append(f"After applying all operations in order, what is the representative (root) of "
                     f"element {metadata.query}? Use union by rank: the root with the higher rank "
                     f"becomes the parent; if the two roots have equal rank, the {tie} becomes the "
                     "parent and the rank of the chosen root increases by one.")
        lines.append("")
        lines.append("The answer is a single integer.")
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        import ast
        try:
            val = ast.literal_eval(answer.strip())
        except Exception:
            return 0.0
        if isinstance(val, bool) or not isinstance(val, int):
            return 0.0
        if int(val) == int(entry.answer):
            return 1.0
        return 0.0
