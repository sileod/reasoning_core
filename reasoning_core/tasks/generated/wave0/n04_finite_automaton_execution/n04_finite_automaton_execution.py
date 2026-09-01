import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'Add explicit finite-automaton state execution.',
 'hypothesis': 'N4',
 'changes': 'Implement DFA/NFA final-state, acceptance, and first-rejection '
            'queries.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3294920948,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 28,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class FiniteAutomatonConfig(Config):
    n_states: int = 5
    word_len: int = 5
    n_letters: int = 2
    max_out: int = 2

    def apply_difficulty(self, level):
        self.n_states = sround(self.n_states + 2 * level)
        self.word_len = sround(self.word_len + 2 * level)
        self.n_letters = sround(self.n_letters + max(0, level - 2))
        self.max_out = sround(self.max_out + (level > 2))


def _run_nfa_active(table, start, word):
    active = {start}
    for ch in word:
        nxt = set()
        for s in active:
            for d in table[s][ch]:
                nxt.add(d)
        active = nxt
    return active


class FiniteAutomatonExecution(Task):
    summary = "Execute nondeterministic finite automata and report active-set size, accepting-state count, or maximum active accepting state."

    config_cls = FiniteAutomatonConfig

    def generate_entry(self):
        cfg = self.config
        n = cfg.n_states
        letters = range(cfg.n_letters)
        letter_names = [chr(ord("a") + i) for i in letters]

        table = {}
        for s in range(n):
            table[s] = {}
            for ch in letters:
                k = random.randint(1, cfg.max_out)
                table[s][ch] = random.sample(range(n), k)

        start = random.randrange(n)
        accept = set(random.sample(range(n), random.randint(1, max(1, n))))
        word = [random.randrange(cfg.n_letters) for _ in range(cfg.word_len)]

        active = _run_nfa_active(table, start, word)
        accepted = bool(active & accept)
        n_accept_active = len(active & accept)
        word_str = "".join(letter_names[c] for c in word)

        # choose query weighted to spread the answer distribution
        if accepted:
            symbol = random.choices(
                ["final-active-size", "accepting-active-count", "final-accepting-state"],
                weights=[1, 2, 1],
            )[0]
        else:
            symbol = random.choices(["final-active-size", "accepting-active-count"], weights=[1, 2])[0]

        if symbol == "final-active-size":
            answer = len(active)
            query = ("the number of distinct states that are active (reachable) after reading "
                     f"the entire input word {word_str!r}")
            fmt = "a nonnegative integer"
        elif symbol == "accepting-active-count":
            answer = n_accept_active
            query = ("the number of accepting states that are active (reachable) after reading "
                     f"the entire input word {word_str!r}")
            fmt = "a nonnegative integer"
        else:
            answer = max(active & accept)
            query = ("the largest accepting state that is active after reading the entire "
                     f"input word {word_str!r}")
            fmt = "a single state number (integer)"

        transition_lines = []
        for s in range(n):
            parts = []
            for ch in letters:
                dst = sorted(table[s][ch])
                parts.append(f"on {letter_names[ch]} -> {{{', '.join(map(str, dst))}}}")
            transition_lines.append(f"state {s}: " + "; ".join(parts))
        alphabet = "{" + ", ".join(letter_names) + "}"

        metadata = edict({
            "payload": {
                "states": f"states 0..{n - 1}, start state {start}, accepting states "
                          f"{{{', '.join(map(str, sorted(accept)))}}}",
                "alphabet": alphabet,
                "transitions": "\n".join(transition_lines),
                "word": f"the input word is {word_str!r} (length {cfg.word_len})",
            },
            "query": query,
            "fmt": fmt,
        })
        return Entry(metadata=metadata, answer=str(answer))

    def render_prompt(self, metadata):
        payload = render_payload(metadata.payload)
        return f"{payload}\n\nCompute {metadata.query}.\n\nThe answer is {metadata.fmt}."

    def score_answer(self, answer, entry):
        try:
            return float(answer) == float(entry.answer)
        except (TypeError, ValueError):
            return 0.0
