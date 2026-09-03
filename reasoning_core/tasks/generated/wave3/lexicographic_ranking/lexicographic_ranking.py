import math
import random
import string
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround


def _counts(word):
    c = {}
    for ch in word:
        c[ch] = c.get(ch, 0) + 1
    return c


def _total_arrangements(word):
    counts = _counts(word)
    perm = math.factorial(len(word))
    for v in counts.values():
        perm //= math.factorial(v)
    return perm


def rank_multiset(word):
    counts = _counts(word)
    letters = sorted(counts)
    rank = 1
    remaining = len(word)
    for ch in word:
        for smaller in letters:
            if smaller >= ch:
                break
            if counts[smaller] > 0:
                counts[smaller] -= 1
                perm = math.factorial(remaining - 1)
                for v in counts.values():
                    perm //= math.factorial(v)
                rank += perm
                counts[smaller] += 1
        counts[ch] -= 1
        remaining -= 1
    return rank


def word_at_rank(rank, counts):
    counts = dict(counts)
    total = sum(counts.values())
    letters = sorted(counts)
    out = []
    for pos in range(total):
        for ch in letters:
            if counts[ch] == 0:
                continue
            counts[ch] -= 1
            perm = math.factorial(total - pos - 1)
            for v in counts.values():
                perm //= math.factorial(v)
            if rank <= perm:
                out.append(ch)
                break
            rank -= perm
            counts[ch] += 1
    return "".join(out)


@dataclass
class LexicographicRankingConfig(Config):
    min_len: int = 3
    max_len: int = 5
    n_letters: int = 3

    def apply_difficulty(self, level):
        self.min_len = sround(self.min_len + level)
        self.max_len = sround(self.max_len + level)
        self.n_letters = sround(self.n_letters + 1 * level)


class LexicographicRanking(Task):
    config_cls = LexicographicRankingConfig

    def generate_entry(self):
        cfg = self.config
        min_len = int(cfg.min_len)
        max_len = int(cfg.max_len)
        n_letters = int(cfg.n_letters)
        length = random.randint(min_len, max_len)
        alphabet = list(string.ascii_lowercase[:n_letters])

        if n_letters >= length and random.random() < 0.5:
            word = "".join(random.sample(alphabet, k=length))
        else:
            base = random.sample(alphabet, k=min(n_letters, length))
            extra = [random.choice(alphabet) for _ in range(length - len(base))]
            pool = base + extra
            random.shuffle(pool)
            word = "".join(pool)

        total = _total_arrangements(word)
        rank = random.randint(1, total)
        counts = _counts(word)

        if random.random() < 0.5:
            rk = rank_multiset(word)
            metadata = edict({"word": word, "direction": "rank",
                              "letters": sorted(counts), "given": word,
                              "payload": {"Letters": word,
                                          "Question": "Give the one-based rank "
                                                      "of this arrangement among "
                                                      "all distinct arrangements "
                                                      "of its letters, in "
                                                      "lexicographic (dictionary) "
                                                      "order, where the first "
                                                      "distinct arrangement has "
                                                      "rank one. Give the answer "
                                                      "as an integer."}})
            answer = str(rk)
        else:
            target = word_at_rank(rank, counts)
            letters = "".join(ch * n for ch, n in sorted(counts.items()))
            metadata = edict({"word": target, "direction": "unrank",
                              "letters": sorted(counts), "given": rank,
                              "payload": {"Letters": letters,
                                          "Question": "List all distinct "
                                                      "arrangements of the "
                                                      "multiset of letters above "
                                                      "in lexicographic "
                                                      "(dictionary) order. Give "
                                                      "the arrangement that "
                                                      "appears at one-based rank "
                                                      + str(rank) + ". Give the "
                                                      "answer as the arrangement "
                                                      "itself."}})
            answer = target

        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return metadata["payload"]["Letters"] + "\n" + metadata["payload"]["Question"]

    def score_answer(self, answer, entry):
        if entry.answer is None:
            return 0.0
        return 1.0 if str(answer).strip() == str(entry.answer).strip() else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'Add lexicographic ranking and unranking over permutations and '
         'multiset words.',
 'hypothesis': 'S32',
 'changes': 'Ask for the rank of a given arrangement, or the arrangement at a '
            'given rank.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1922532027,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
