import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'trie_unique_prefix (draw 1 of 2)',
 'hypothesis': 'W1-012',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/trie_unique_prefix',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1967867643,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _unique_prefix(words, target):
    """Return the shortest prefix of `target` that no other word starts with."""
    n = len(target)
    for k in range(1, n + 1):
        pref = target[:k]
        if all(not w.startswith(pref) for w in words if w != target):
            return pref
    return target


def _gen_words(n_words, min_len, max_len, alphabet, rng):
    while True:
        words = []
        for _ in range(n_words):
            ln = rng.randint(min_len, max_len)
            words.append(''.join(rng.choice(alphabet) for _ in range(ln)))
        if len(set(words)) == n_words:
            return words


@dataclass
class TrieUniquePrefixConfig(Config):
    n_words: int = 4
    min_len: int = 3
    max_len: int = 6
    alphabet_size: int = 4

    def apply_difficulty(self, level):
        self.n_words = sround(self.n_words + level * 2)
        self.max_len = sround(self.max_len + level)
        self.min_len = 3
        self.alphabet_size = sround(max(2, self.alphabet_size - level // 2))


class TrieUniquePrefix(Task):
    summary = "Given a list of strings, output the shortest prefix that uniquely identifies the indicated target among the others."
    config_cls = TrieUniquePrefixConfig

    def generate_entry(self):
        c = self.config
        alphabet = "abcdefghijklmnopqrstuvwxyz"[:c.alphabet_size]
        target_idx = random.randrange(c.n_words)
        for _ in range(400):
            words = _gen_words(c.n_words, c.min_len, c.max_len, alphabet, random)
            target = words[target_idx]
            answer = _unique_prefix(words, target)
            # valid iff exactly the target matches the answer prefix among all words
            hits = [w.startswith(answer) for w in words]
            if sum(hits) == 1 and hits[target_idx]:
                break
        else:
            raise RuntimeError("could not construct a unique-prefix instance")

        metadata = edict({
            "words": list(words),
            "target": target,
            "target_index": int(target_idx),
            "answer": answer,
        })
        metadata.payload = {"words": list(words), "target": target}
        # verify
        assert _unique_prefix(list(words), target) == answer
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        payload = {"words": metadata.words, "target": metadata.target}
        return (
            f"{render_payload(payload)}\n\n"
            "Give the shortest prefix (starting from the first character) of the target "
            "string that is not a prefix of any other string in the list. "
            "The prefix must begin at the target's first character and be as short as possible "
            "while still matching only the target. Answer with exactly that prefix string."
        )

    def score_answer(self, answer, entry):
        return 1.0 if isinstance(answer, str) and answer == entry.answer else 0.0
