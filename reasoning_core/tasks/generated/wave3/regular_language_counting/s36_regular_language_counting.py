import random
from dataclasses import dataclass

from automata.fa.nfa import NFA
from reasoning_core.template import Task, Entry, Config, edict
from reasoning_core.utils import score_scalar

_ALPHABETS = {
    2: ["a", "b"],
    3: ["a", "b", "c"],
}


def _rand_letter(size):
    return random.choice(_ALPHABETS[size])


def _rand_class(size):
    letters = _ALPHABETS[size]
    return "".join(sorted(random.sample(letters, random.randint(1, len(letters)))))


def _gen_unit(size):
    letters = _ALPHABETS[size]
    r = random.random()
    if r < 0.35:
        letter = random.choice(letters)
        return f"{letter}{random.choice(['*', '+'])}"
    if r < 0.7:
        cls = _rand_class(size)
        return f"[{cls}]{random.choice(['*', '+', '?'])}"
    if r < 0.9:
        letter = random.choice(letters)
        return f"({letter}){random.choice(['*', '+', '?'])}"
    cls = _rand_class(size)
    return f"[{cls}]"


def _rand_pattern(size):
    n = random.randint(1, 3)
    return "".join(_gen_unit(size) for _ in range(n))


def _count_matches(pattern, length, alphabet):
    """Count distinct strings of given length matched by the regex.

    Builds an NFA via Thompson construction, eliminates epsilon transitions,
    determinises via subset construction, then counts accepting paths by DP.
    """
    symbols = set(alphabet) | {""}
    nfa = NFA.from_regex(pattern, input_symbols=symbols)

    nfa_states = list(nfa.states)
    trans = {}
    for s in nfa_states:
        d = {}
        for sym, targets in nfa.transitions[s].items():
            d.setdefault(sym, set()).update(targets)
        trans[s] = d
    init = nfa.initial_state
    finals = set(nfa.final_states)

    def closure(states):
        stack = list(states)
        cl = set(states)
        while stack:
            s = stack.pop()
            for t in trans[s].get("", ()):
                if t not in cl:
                    cl.add(t)
                    stack.append(t)
        return cl

    start = frozenset(closure([init]))

    dfa = {}
    dfa_finals = set()
    queue = [start]
    seen = {start}
    while queue:
        cur = queue.pop(0)
        accepting = any(s in finals for s in cur)
        if accepting:
            dfa_finals.add(cur)
        dfa[cur] = {}
        for sym in alphabet:
            nxt = set()
            for s in cur:
                for t in trans[s].get(sym, ()):
                    nxt.add(t)
            nxt = frozenset(closure(nxt))
            if not nxt:
                continue
            dfa[cur][sym] = nxt
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)

    dp = {start: 1}
    for _ in range(length):
        ndp = {}
        for st, cnt in dp.items():
            for nxt in dfa[st].values():
                ndp[nxt] = ndp.get(nxt, 0) + cnt
        dp = ndp

    total = 0
    for st, cnt in dp.items():
        if st in dfa_finals:
            total += cnt
    return total


@dataclass
class RegularLanguageCountingConfig(Config):
    alphabet_size: int = 2
    length_min: int = 6
    length_max: int = 8

    def apply_difficulty(self, level):
        self.alphabet_size = 2 if level < 3 else 3
        self.length_min = 5 + level
        self.length_max = 8 + level


class RegularLanguageCounting(Task):
    config_cls = RegularLanguageCountingConfig

    def __init__(self, seed=None):
        super().__init__(seed=seed)

    def generate_entry(self):
        alphabet_size = self.config.alphabet_size
        alphabet = _ALPHABETS[alphabet_size]

        while True:
            pattern = _rand_pattern(alphabet_size)
            length = random.randint(self.config.length_min, self.config.length_max)
            total = alphabet_size ** length
            count = _count_matches(pattern, length, alphabet)
            if count >= 5 and count != total and count != length:
                break

        metadata = edict(
            {
                "pattern": pattern,
                "length": int(length),
                "alphabet": list(alphabet),
                "count": int(count),
            }
        )
        metadata.payload = {
            "pattern": pattern,
            "length": int(length),
            "alphabet": list(alphabet),
        }
        return Entry(metadata=metadata, answer=str(count))

    def render_prompt(self, metadata):
        alphabet = metadata.payload["alphabet"]
        letters = ", ".join(repr(c) for c in alphabet)
        return (
            f"The alphabet is {{{letters}}}. How many distinct strings of "
            f"exactly {metadata.payload['length']} characters match the regular "
            f"expression {metadata.payload['pattern']}? "
            f"The answer is a non-negative integer."
        )

    def score_answer(self, answer, entry):
        return score_scalar(answer, entry)


TASK_META = {'parent_source_id': None,
 'idea': 'Add counting over the language of a small regular expression.',
 'hypothesis': 'S36',
 'changes': 'Ask how many strings of a stated length the described pattern '
            'matches.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 168153944,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
