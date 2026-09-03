import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'suffix_array_rank (draw 1 of 2)',
 'hypothesis': 'W1-071',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/suffix_array_rank',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3317076631,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def build_suffix_array(s):
    n = len(s)
    sa = list(range(n))
    rank = list(map(ord, s))
    k = 1
    tmp = [0] * n
    while True:
        sa.sort(key=lambda x: (rank[x], rank[x + k] if x + k < n else -1))
        tmp[sa[0]] = 0
        for i in range(1, n):
            prev, cur = sa[i - 1], sa[i]
            prev_pair = (rank[prev], rank[prev + k] if prev + k < n else -1)
            cur_pair = (rank[cur], rank[cur + k] if cur + k < n else -1)
            tmp[cur] = tmp[prev] + (1 if prev_pair != cur_pair else 0)
        rank, tmp = tmp, rank
        if rank[sa[-1]] == n - 1:
            break
        k <<= 1
    return sa


@dataclass
class SuffixRankConfig(Config):
    min_len: int = 6
    max_len: int = 14
    alpha: int = 4

    def apply_difficulty(self, level):
        self.min_len = sround(self.min_len + level)
        self.max_len = sround(self.max_len + level + 1)
        self.alpha = self.alpha if level < 3 else self.alpha + 2


class SuffixRank(Task):
    task_name = "suffix_array_rank"
    summary = "Given a string and a suffix start index, output that suffix's integer rank (0-based) in the suffix array, over varied alphabet and length strings."

    config_cls = SuffixRankConfig

    def generate_entry(self):
        cfg = self.config
        alphabet_start = 97
        while True:
            length = random.randint(cfg.min_len, cfg.max_len)
            alpha = cfg.alpha
            chars = [chr(alphabet_start + random.randrange(alpha)) for _ in range(length)]
            s = "".join(chars)
            if not s:
                continue
            sa = build_suffix_array(s)
            idx = random.randrange(length)
            sort_key = s[idx:]
            ordered = sorted(s[i:] for i in range(length))
            rank = ordered.index(sort_key)
            assert rank == sa.index(idx)
            break
        metadata = edict()
        metadata.payload = {"s": s, "index": idx}
        metadata.s = s
        metadata.index = idx
        metadata.rank = rank
        return Entry(metadata=metadata, answer=str(rank))

    def render_prompt(self, metadata):
        return (
            "A suffix of a string is the substring that starts at some position and runs to the end. "
            "The suffix array of a string lists its suffixes in lexicographic (dictionary) order. "
            "For the string {s}, consider the suffix that starts at index {index} "
            "(0-based, so index 0 is the whole string). "
            "What is the 0-based rank of this suffix in the suffix array of \"{s}\"? "
            "Rank 0 means the suffix is the smallest in lexicographic order. "
            "The answer is an integer: the 0-based rank. Write only the number."
        ).format(s=metadata.s, index=metadata.index)

    def score_answer(self, answer, entry):
        try:
            val = answer.strip()
            parsed = int(val)
        except Exception:
            return 0.0
        return 1.0 if parsed == int(entry.answer) else 0.0
