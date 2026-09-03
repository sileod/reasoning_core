import random
from dataclasses import dataclass

from reasoning_core.template import Entry, Config, Task, edict, stochastic_rounding as sround

_RULES = ("take", "square", "split")


@dataclass
class GrundyValuesConfig(Config):
    n_heaps: int = 2
    max_size: int = 6
    n_distinct_rules: int = 1
    max_attempts: int = 200

    def apply_difficulty(self, level):
        self.n_heaps = max(2, min(5, sround(self.n_heaps + 0.45 * level)))
        self.max_size = sround(self.max_size + 1.2 * level)
        self.n_distinct_rules = max(1, min(3, sround(self.n_distinct_rules + 0.55 * level)))


def _grundy_table(rule, max_size):
    g = [0] * (max_size + 1)
    for n in range(1, max_size + 1):
        if rule == "take":
            moves = [g[n - k] for k in (1, 2, 3) if n - k >= 0]
        elif rule == "square":
            moves = [g[n - k * k] for k in range(1, int(n**0.5) + 1)]
        elif rule == "split":
            moves = [g[i] ^ g[n - i] for i in range(1, n) if i < n - i]
        else:
            raise ValueError(rule)
        present = set(moves)
        mex = 0
        while mex in present:
            mex += 1
        g[n] = mex
    return g


_RULE_WORDS = {
    "take": "remove any one, two, or three tokens",
    "square": "remove any square number of tokens (1, 4, 9, ...)",
    "split": "split the heap into two non-empty heaps of unequal sizes, replacing it by those two",
}

TASK_META = {'parent_source_id': None,
 'idea': 'Add Sprague-Grundy reasoning over a sum of small impartial games.',
 'hypothesis': 'S31',
 'changes': 'Ask who wins a described position, or the Grundy value of the '
            'whole game.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3693162845,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


class GrundyValues(Task):
    summary = "Report the Sprague-Grundy value of a position made of independent heaps."
    config_cls = GrundyValuesConfig

    def generate_entry(self):
        cfg = self.config
        for _ in range(cfg.max_attempts):
            n_heaps = cfg.n_heaps
            n_distinct = max(1, min(3, cfg.n_distinct_rules, n_heaps))
            distinct = random.sample(_RULES, n_distinct)
            sizes = [random.randint(2, cfg.max_size) for _ in range(n_heaps)]
            tables = {r: _grundy_table(r, cfg.max_size) for r in distinct}
            heaps = []
            for i in range(n_heaps):
                rule = distinct[i % n_distinct]
                size = sizes[i]
                heaps.append((size, rule))
            whole = 0
            per = []
            for size, rule in heaps:
                g = tables[rule][size]
                whole ^= g
                per.append(g)
            answer = " ".join(str(v) for v in per) + " " + str(whole)
            metadata = edict(
                heaps=heaps,
                per_heap=per,
                whole=whole,
                distinct_rules=distinct,
                payload={"heaps": [(s, r) for s, r in heaps]},
            )
            return Entry(metadata=metadata, answer=answer)
        raise RuntimeError("Failed to generate a nontrivial Grundy position")

    def render_prompt(self, metadata):
        labels = ["A", "B", "C", "D", "E"]
        lines = []
        for idx, (size, rule) in enumerate(metadata.heaps):
            lines.append(
                f"Heap {labels[idx]} has {size} tokens; its only legal moves {_RULE_WORDS[rule]}."
            )
        head = "Consider the following impartial game made of independent heaps. A player on their turn moves in exactly one heap. The player unable to move loses."
        n = len(metadata.heaps)
        feat = " ".join(labels[i] for i in range(n))
        return (
            head
            + "\n"
            + "\n".join(lines)
            + f"\nGive each heap's Grundy value, in the heap order {feat}, followed by the Grundy value of the whole position. The answer is {n + 1} space-separated non-negative integers."
        )

    def score_answer(self, answer, entry):
        try:
            got = [int(x) for x in str(answer).split()]
        except (ValueError, TypeError):
            return 0.0
        want = [int(x) for x in entry.answer.split()]
        return 1.0 if got == want else 0.0
