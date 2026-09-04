"""Determinize finite automata by epsilon closure and subset construction."""

import random
from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, edict, render_payload

TASK_META = {'parent_source_id': None,
 'idea': 'nfa_subset_construction (draw 1 of 1)',
 'hypothesis': 'HV-057',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/nfa_subset_construction',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 929441144,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _epsilon_closure(state, epsilon, n):
    reach = {state}
    stack = [state]
    while stack:
        s = stack.pop()
        for t in epsilon.get(s, ()):
            if t not in reach:
                reach.add(t)
                stack.append(t)
    return frozenset(sorted(reach))


def _closure_of_set(states, epsilon, n):
    out = set()
    for s in states:
        out |= set(_epsilon_closure(s, epsilon, n))
    return frozenset(sorted(out))


def _determinize(n, alphabet, delta, epsilon, start, accepts):
    start_closure = _closure_of_set({start}, epsilon, n)
    dfa = {}                    # subset -> {symbol: subset}
    order = []
    queue = [start_closure]
    seen = {start_closure}
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        dfa[cur] = {}
        for sym in alphabet:
            move = set()
            for s in cur:
                for t in delta.get((s, sym), ()):
                    move.add(t)
            if not move:
                continue
            nxt = _closure_of_set(move, epsilon, n)
            dfa[cur][sym] = nxt
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return dfa, order, seen


def _state_name(subset):
    return "{" + ",".join(map(str, subset)) + "}"


@dataclass
class NfaSubsetConstructionConfig(Config):
    n_states: int = 5
    alphabet_size: int = 2
    edge_density: int = 2
    eps_density: int = 1

    def apply_difficulty(self, level):
        self.n_states = int(5 + level)
        self.alphabet_size = int(2 + (level >= 3))
        self.edge_density = 2 + level
        self.eps_density = 1 + (level >= 2)


class NfaSubsetConstruction(Task):
    summary = ("Determinize finite automata by epsilon closure and subset "
               "construction, returning the number of reachable canonical DFA "
               "states (an integer) across alphabets of size one and two, with "
               "and without epsilon transitions, at growing state counts.")
    config_cls = NfaSubsetConstructionConfig
    task_version = 2

    def generate_entry(self):
        n = self.config.n_states
        alphabet = list(range(self.config.alphabet_size))
        start = 0

        lo_floor = max(2, round(0.6 * n))
        hi_floor = max(lo_floor + 1, round(0.9 * n) + 1)

        order = []
        attempts = 0
        while True:
            attempts += 1
            floor = random.randint(lo_floor, hi_floor)
            delta = {}
            for s in range(n):
                for _ in range(self.config.edge_density):
                    sym = random.choice(alphabet)
                    t = random.randrange(n)
                    delta.setdefault((s, sym), [])
                    if t not in delta[(s, sym)]:
                        delta[(s, sym)].append(t)
            epsilon = {}
            for s in range(n):
                for _ in range(self.config.eps_density * 2):
                    if random.random() < 0.5:
                        t = random.randrange(n)
                        if t != s:
                            epsilon.setdefault(s, [])
                            if t not in epsilon[s]:
                                epsilon[s].append(t)

            accepts = set(random.sample(range(n), max(1, n // 2)))

            _dfa, order, _seen = _determinize(n, alphabet, delta, epsilon, start, accepts)

            if len(order) >= floor or attempts >= 400:
                break

        answer = str(len(order))

        record = []
        for s in range(n):
            d = {}
            for sym in alphabet:
                d[str(sym)] = sorted(delta.get((s, sym), ()))
            e = sorted(epsilon.get(s, ()))
            record.append({"trans": d, "epsilon": e})
        metadata = edict({
            "n_states": n,
            "alphabet": [str(x) for x in alphabet],
            "delta": record,
            "accepting": sorted(accepts),
            "start": start,
        })
        metadata.payload = {
            "n_states": n,
            "alphabet": [str(x) for x in alphabet],
            "delta": record,
            "accepting": sorted(accepts),
            "start": start,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = []
        lines.append("A nondeterministic finite automaton (NFA) has states "
                     "0..{} and alphabet {}.".format(
                         metadata.n_states - 1,
                         ",".join(metadata.alphabet)))
        lines.append("Its transition relation and epsilon transitions (moves "
                     "possible without consuming a symbol) are:")
        for s in range(metadata.n_states):
            deli = metadata.delta[s]
            eps = deli.get("epsilon", [])
            parts = []
            for sym, targets in deli.get("trans", {}).items():
                if targets:
                    parts.append("on {} go to {}".format(sym, ",".join(map(str, targets))))
            eps_str = ",".join(map(str, eps)) if eps else "none"
            line = "From state {}: {}; epsilon to {}.".format(
                s, "; ".join(parts) if parts else "no transitions", eps_str)
            lines.append(line)
        lines.append("The initial state is {} and the accepting states are {}."
                     .format(metadata.start,
                             ",".join(map(str, metadata.accepting))))
        lines.append("Apply the subset construction (with epsilon closure to "
                     "take unreachable-at-epsilon states into account) to "
                     "convert this NFA into a deterministic finite automaton "
                     "(DFA) whose states are the reachable canonical subsets "
                     "of NFA states. A DFA only has a transition when the "
                     "source subset has at least one outgoing move on that "
                     "symbol.")
        lines.append("How many reachable states does the resulting DFA have?")
        lines.append("The answer is the number of reachable DFA states, given "
                     "as a single integer.")
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        gt = entry.answer
        s = str(answer).strip()
        try:
            v = int(s)
        except (ValueError, TypeError):
            return 0.0
        return 1.0 if str(v) == gt else 0.0
