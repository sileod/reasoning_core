"""Execute LRU, LFU, or FIFO cache accesses with inserts and evictions under
explicit tie rules, returning hits, misses, or final cache state."""

from dataclasses import dataclass

import random

from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'cache_replacement_trace (draw 1 of 1)',
 'hypothesis': 'HV-041',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/cache_replacement_trace',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1492106328,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _simulate(policy, capacity, accesses, tie):
    """Simulate the cache. Returns (hits, misses, sorted final cache keys).

    policy is 'LRU', 'LFU' or 'FIFO'. tie only matters for LFU and is one of
    'lfu_insertion' (break frequency ties by earliest insertion) or 'lfu_key'
    (break frequency ties by smallest key).
    """
    cache = set()
    hits = 0
    misses = 0
    order = []          # FIFO: insertion order (oldest first)
    recency = []        # LRU: most recent at end
    freq = {}           # LFU: access count per resident key
    ins_idx = {}        # LFU: insertion sequence number per resident key
    counter = 0
    for key in accesses:
        if key in cache:
            hits += 1
            if policy == 'LRU':
                recency.remove(key)
                recency.append(key)
            elif policy == 'LFU':
                freq[key] += 1
        else:
            misses += 1
            if len(cache) < capacity:
                cache.add(key)
                if policy == 'FIFO':
                    order.append(key)
                elif policy == 'LRU':
                    recency.append(key)
                elif policy == 'LFU':
                    freq[key] = 1
                    ins_idx[key] = counter
                    counter += 1
            else:
                victim = _evict(policy, cache, order, recency, freq, ins_idx, tie)
                cache.discard(victim)
                if policy == 'FIFO':
                    order.remove(victim)
                    order.append(key)
                elif policy == 'LRU':
                    recency.remove(victim)
                    recency.append(key)
                elif policy == 'LFU':
                    del freq[victim]
                    del ins_idx[victim]
                    freq[key] = 1
                    ins_idx[key] = counter
                    counter += 1
                cache.add(key)
    return hits, misses, sorted(cache)


def _evict(policy, cache, order, recency, freq, ins_idx, tie):
    if policy == 'FIFO':
        return order[0]
    if policy == 'LRU':
        return recency[0]
    # LFU: fewest accesses; tie-break per tie rule
    if tie == 'lfu_key':
        return min(freq, key=lambda k: (freq[k], k))
    return min(freq, key=lambda k: (freq[k], ins_idx[k], k))


@dataclass
class CacheTraceConfig(Config):
    capacity_base: int = 3
    length_base: int = 8
    alphabet_base: int = 6

    def apply_difficulty(self, level):
        self.capacity_base = sround(self.capacity_base + level)
        self.length_base = sround(self.length_base + 3 * level)
        self.alphabet_base = sround(self.alphabet_base + 2 * level)


def _tie_description(policy, tie):
    if policy == 'FIFO':
        return ("The evicted slot is the key that has been in the cache the longest, "
                "i.e. the earliest-inserted resident key.")
    if policy == 'LRU':
        return ("The evicted slot is the key that has not been accessed for the longest "
                "time, i.e. the least-recently-used resident key.")
    if tie == 'lfu_key':
        return ("The evicted slot is the resident key with the fewest accesses since it "
                "was inserted; ties are broken by the smallest key.")
    return ("The evicted slot is the resident key with the fewest accesses since it was "
            "inserted; ties are broken by the earliest insertion.")


class CacheReplacementTrace(Task):
    summary = ("Execute LRU, LFU, or FIFO cache accesses with inserts and evictions under "
               "explicit tie rules, returning hits, misses, or final cache state. Instance "
               "family: randomized access traces over integer keys with varying capacity, "
               "alphabet and length; modes: LRU (least-recently-used), LFU (fewest accesses "
               "with insertion-first or key-first tie-break), FIFO (earliest-inserted). "
               "Output regimes: hit count, miss count, or sorted final cache state; answers "
               "are compact integers or sorted comma-separated lists.")
    config_cls = CacheTraceConfig
    task_version = 2

    def generate_entry(self):
        cfg = self.config
        cap = max(1, cfg.capacity_base)
        length = max(1, cfg.length_base)
        alphabet = max(cap, cfg.alphabet_base)
        policy = random.choice(['LRU', 'LFU', 'FIFO'])
        if policy == 'LFU':
            tie = random.choice(['lfu_insertion', 'lfu_key'])
        else:
            tie = random.choice(['lfu_insertion', 'lfu_key'])
        keys = list(range(alphabet))
        accesses = [random.choice(keys) for _ in range(length)]
        hits, misses, final_state = _simulate(policy, cap, accesses, tie)
        assert hits + misses == length, (hits, misses, length)
        assert 0 <= hits <= length
        assert len(final_state) <= cap
        query = random.choice(['hits', 'misses', 'state'])
        metadata = edict({
            'policy': policy,
            'capacity': cap,
            'alphabet': alphabet,
            'length': length,
            'tie': tie,
            'accesses': accesses,
            'hits': hits,
            'misses': misses,
            'final_state': final_state,
            'query': query,
        })
        metadata.payload = {
            'policy': policy, 'capacity': cap, 'alphabet': alphabet,
            'accesses': accesses, 'tie': tie, 'query': query,
        }
        return Entry(metadata=metadata, answer=self._format_answer(metadata))

    def _format_answer(self, m):
        if m.query == 'hits':
            return str(m.hits)
        if m.query == 'misses':
            return str(m.misses)
        return '[' + ', '.join(str(k) for k in m.final_state) + ']'

    def render_prompt(self, metadata):
        query_desc = {
            'hits': 'How many of the accesses are hits?',
            'misses': 'How many of the accesses are misses?',
            'state': ('What is the final set of keys in the cache, written as a '
                      'comma-separated sorted list inside square brackets?'),
        }[metadata.query]
        lines = [
            f"A cache holds at most {metadata.capacity} distinct keys, drawn from the "
            f"alphabet {{0, 1, ..., {metadata.alphabet - 1}}}.",
            f"It uses {metadata.policy} replacement. "
            + _tie_description(metadata.policy, metadata.tie),
            "The cache starts empty. Process this sequence of key accesses in order: "
            + ' '.join(str(k) for k in metadata.accesses),
            query_desc,
        ]
        return ' '.join(lines)

    def score_answer(self, answer, entry):
        gold = entry.answer
        a = answer.strip()
        if entry.metadata['query'] == 'state':
            return 1.0 if _parse_state(a) == _parse_state(gold) else 0.0
        return 1.0 if a == gold.strip() else 0.0


def _parse_state(s):
    s = s.strip()
    if not (s.startswith('[') and s.endswith(']')):
        return None
    inner = s[1:-1].strip()
    if inner == "":
        return []
    try:
        parts = [int(x) for x in inner.replace(',', ' ').split()]
    except ValueError:
        return None
    return sorted(parts)
