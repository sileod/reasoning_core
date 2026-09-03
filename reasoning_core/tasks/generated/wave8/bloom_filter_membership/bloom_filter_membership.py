from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


@dataclass
class BloomFilterConfig(Config):
    m_bits: int = 24
    k_hashes: int = 2
    n_items: int = 4
    t_queries: int = 5

    def apply_difficulty(self, level):
        self.m_bits = sround(self.m_bits + 8 * level)
        self.k_hashes = sround(self.k_hashes + level)
        self.n_items = sround(self.n_items + 2 * level)
        self.t_queries = sround(self.t_queries + level)


def _parse_indices(answer):
    if not isinstance(answer, str):
        return None
    a = answer.strip().replace(",", " ")
    if a == "":
        return []
    try:
        vals = [int(tok) for tok in a.split()]
    except ValueError:
        return None
    if any(v < 0 for v in vals):
        return None
    return sorted(set(vals))


def _score(answer, entry):
    gold = _parse_indices(entry.answer)
    pred = _parse_indices(answer)
    if pred is None:
        return 0.0
    return 1.0 if pred == gold else 0.0


class BloomFilterMembership(Task):
    summary = (
        "Apply explicit linear-mod hashes to a Bloom filter bit array and classify each of "
        "several query items as definitely absent or possibly present, emitting the sorted "
        "indices of the definitely-absent queries."
    )
    config_cls = BloomFilterConfig

    def generate_entry(self):
        import random

        c = self.config
        m = c.m_bits
        k = c.k_hashes
        n = c.n_items
        t = c.t_queries

        for _ in range(300):
            a = [random.randint(1, m - 1) for _ in range(k)]
            b = [random.randint(0, m - 1) for _ in range(k)]

            def hashes(x, _a=a, _b=b):
                return [(_a[j] * x + _b[j]) % m for j in range(k)]

            inserted = random.sample(range(0, 1000), n)
            bits = [0] * m
            for x in inserted:
                for p in hashes(x):
                    bits[p] = 1

            inserted_set = set(inserted)
            pool = [v for v in range(0, 1000) if v not in inserted_set]

            abs_idx = []
            for _ in range(500):
                queries = random.sample(pool, t)
                abs_idx = [i for i, q in enumerate(queries)
                           if any(bits[p] == 0 for p in hashes(q))]
                if 1 <= len(abs_idx) <= t - 1:
                    break
            else:
                continue
            break
        else:
            raise RuntimeError("could not balance bloom filter membership")

        gold = sorted(abs_idx)
        bitstring = "".join(str(bv) for bv in bits)

        metadata = edict({
            "inserted": inserted,
            "queries": queries,
            "a": a,
            "b": b,
            "bits": bitstring,
        })
        metadata.payload = {
            "m": int(m),
            "k": int(k),
            "a": [int(v) for v in a],
            "b": [int(v) for v in b],
            "bits": bitstring,
            "queries": [int(q) for q in queries],
        }
        return Entry(metadata=metadata, answer=" ".join(str(i) for i in gold))

    def render_prompt(self, metadata):
        payload = render_payload(metadata.payload)
        return (
            "An m-bit Bloom filter has all m bit positions 0..m-1 initialized to 0. Its k hash "
            "functions map an item x to a bit index by h_j(x) = (a_j*x + b_j) mod m. The "
            f"parameters are: {payload} (in the 'bits' string, position 0 is the leftmost bit). "
            "A set of items was inserted by setting, for every item, each of its k hash "
            "positions to 1. Query the items listed under 'queries'. An item is DEFINITELY "
            "ABSENT if at least one of its k hash positions is still 0; otherwise it is only "
            "POSSIBLY PRESENT. Give your answer as the 0-based indices (in the order the items "
            "appear under 'queries') of the items that are DEFINITELY ABSENT, formatted as a "
            "space-separated list sorted in ascending order; if queries 1 and 3 were absent, "
            "answer: 1 3. Write only the space-separated indices."
        )

    def score_answer(self, answer, entry):
        return _score(answer, entry)


TASK_META = {'parent_source_id': None,
 'idea': 'bloom_filter_membership (draw 1 of 2)',
 'hypothesis': 'W1-011',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/bloom_filter_membership',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 244270231,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
