import random
from collections import deque
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict


TASK_META = {'parent_source_id': None,
 'idea': 'Add a task that separates two finite automata by exhibiting a '
         'witness string.',
 'hypothesis': 'S38',
 'changes': 'Give two small automata or two regular expressions and ask for '
            'the shortest string accepted by one and rejected by the other.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1931387801,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _render_transition_table(n_states, alphabet, table):
    header = "   | " + " ".join(c for c in alphabet)
    line = "---+-" + "----" * len(alphabet)
    lines = [header, line]
    for s in range(n_states):
        cells = " ".join(str(table[s][i]) for i in range(len(alphabet)))
        lines.append("  %d | %s" % (s, cells))
    return "\n".join(lines)


@dataclass
class LanguageSeparationConfig(Config):
    n_states: int = 3
    alphabet_size: int = 2
    none_fraction: float = 0.15

    def apply_difficulty(self, level):
        self.n_states = int(3 + level)
        self.alphabet_size = int(2 + (1 if level >= 3 else 0))


def _build_one(n_states, alphabet, rng):
    start = rng.randrange(n_states)
    acc = set()
    for s in range(n_states):
        if rng.random() < 0.4:
            acc.add(s)
    table = []
    for s in range(n_states):
        row = [rng.randrange(n_states) for _ in alphabet]
        table.append(row)
    return start, acc, table


def _shortest_witness(alphabet, s1, acc1, t1, s2, acc2, t2):
    start = (s1, s2)
    if s1 in acc1 and s2 not in acc2:
        return ""
    seen = {start}
    dq = deque([(start, "")])
    while dq:
        (p, q), w = dq.popleft()
        for i, c in enumerate(alphabet):
            np = t1[p][i]
            nq = t2[q][i]
            if np in acc1 and nq not in acc2:
                return w + c
            nstate = (np, nq)
            if nstate not in seen:
                seen.add(nstate)
                dq.append((nstate, w + c))
    return None


class LanguageSeparation(Task):
    config_cls = LanguageSeparationConfig

    def generate_entry(self):
        n_states = int(self.config.n_states)
        alph_size = int(self.config.alphabet_size)
        alphabet = [chr(ord('a') + i) for i in range(alph_size)]

        if random.random() < self.config.none_fraction:
            s1, acc1, t1 = _build_one(n_states, alphabet, random)
            s2, acc2, t2 = s1, set(acc1), [row[:] for row in t1]
            witness = None
        else:
            witness = None
            tries = 0
            best_len = -1
            best_spec = None
            while tries < 600:
                cand1 = _build_one(n_states, alphabet, random)
                cand2 = _build_one(n_states, alphabet, random)
                w = _shortest_witness(alphabet, cand1[0], cand1[1], cand1[2],
                                      cand2[0], cand2[1], cand2[2])
                if w is not None and len(w) >= 2:
                    witness = w
                    s1, acc1, t1 = cand1
                    s2, acc2, t2 = cand2
                    break
                if w is not None and len(w) > best_len:
                    best_len = len(w)
                    best_spec = (cand1, cand2, w)
                tries += 1
            if witness is None:
                if best_spec is None:
                    raise RuntimeError("could not find witness after resampling")
                cand1, cand2, witness = best_spec
                s1, acc1, t1 = cand1
                s2, acc2, t2 = cand2

        meta = edict({
            "alphabet": alphabet,
            "n1": n_states, "n2": n_states,
            "start1": int(s1), "start2": int(s2),
            "acc1": [int(x) for x in sorted(acc1)],
            "acc2": [int(x) for x in sorted(acc2)],
            "table1": [[int(x) for x in row] for row in t1],
            "table2": [[int(x) for x in row] for row in t2],
            "witness": witness,
        })
        meta.payload = {
            "alphabet": alphabet,
            "automaton1": self._describe(n_states, alphabet, s1, acc1, t1),
            "automaton2": self._describe(n_states, alphabet, s2, acc2, t2),
        }
        answer = witness if witness is not None else "none"
        return Entry(metadata=meta, answer=answer)

    def _describe(self, n_states, alphabet, start, acc, table):
        lines = []
        lines.append("States: " + ", ".join(str(i) for i in range(n_states)))
        lines.append("Alphabet: " + ", ".join(alphabet))
        lines.append("Start state: " + str(start))
        lines.append("Accepting states: " +
                     (", ".join(str(i) for i in sorted(acc)) if acc else "none"))
        lines.append("Transition table (row i, column labeled by a symbol, "
                     "entry = next state):")
        lines.append(_render_transition_table(n_states, alphabet, table))
        return "\n".join(lines)

    def render_prompt(self, metadata):
        return (
            "Two deterministic finite automata are given.\n\n"
            "Automaton 1:\n" + metadata.payload["automaton1"] + "\n\n"
            "Automaton 2:\n" + metadata.payload["automaton2"] + "\n\n"
            "Find the lexicographically smallest string of shortest length "
            "that is accepted by Automaton 1 and rejected by Automaton 2. "
            "If the two automata accept exactly the same set of strings and "
            "no such string exists, answer \"none\". "
            "The answer is the witness string, or \"none\"."
        )

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        if not isinstance(answer, str):
            return 0.0
        return 1.0 if answer == entry.answer else 0.0
