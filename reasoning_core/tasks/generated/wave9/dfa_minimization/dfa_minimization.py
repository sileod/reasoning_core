import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'dfa_minimization (draw 1 of 1)',
 'hypothesis': 'HV-058',
 'changes': 'new task in reasoning_core/tasks/generated/wave9/dfa_minimization',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1633967595,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class DFAConfig(Config):
    min_states: int = 4
    max_states: int = 7
    alphabet_size: int = 2

    def apply_difficulty(self, level):
        self.min_states = sround(self.min_states + level)
        self.max_states = sround(self.max_states + level * 2)
        self.alphabet_size = sround(self.alphabet_size + (level // 3))
        if self.min_states > self.max_states:
            self.min_states = self.max_states


def _minimize(states, alphabet, trans, accepting):
    """Return the canonical partition. states is a list of state indices in order."""
    n = len(states)
    # partition refinement (Moore's algorithm)
    labels = [1 if s in accepting else 0 for s in states]
    while True:
        # signature: (label, tuple of labels of successors)
        sig = [(labels[i], tuple(labels[int(trans[i][c])] for c in range(len(alphabet))))
               for i in range(n)]
        new_labels = {}
        for s in sorted(set(sig), key=str):
            new_labels[s] = len(new_labels)
        new = [new_labels[s] for s in sig]
        if new == labels:
            break
        labels = new
    return labels


def _canonical_block(labels):
    """Return representative classes, i.e. sorted list mapping each original state to its block."""
    # relabel blocks 0..k-1 in order of first appearance of original states
    order = {}
    for idx in range(len(labels)):
        b = labels[idx]
        if b not in order:
            order[b] = len(order)
    block_of = [order[b] for b in labels]
    return block_of


def generate_one(max_states, n_min, alphabet, randomizer):
    """Build a DFA with a guaranteed nontrivial amount of merging.

    Strategy: partition the n states into blocks; make some pairs behave
    identically (same accepting status and identical transitions block-wise),
    while other transitions are scrambled so that the refinement only merges
    the blocks the generator intends. We then verify with the independent
    Moore refinement that the final canonical block count matches the intended
    number of distinct blocks.
    """
    symbols = [chr(ord('a') + c) for c in range(len(alphabet))]
    while True:
        n = randomizer.randint(min(n_min, max_states), max_states)
        # choose a target number of equivalence blocks, less than n but >= 2
        nblocks = randomizer.randint(2, max(2, n - 1))
        # assign each state to a block; we want several states sharing blocks
        block_assign = [randomizer.randrange(nblocks) for _ in range(n)]
        # ensure every block is non-empty
        if len(set(block_assign)) != nblocks:
            continue
        # ensure at least one block has size >= 2 (real merging)
        sizes = {}
        for b in block_assign:
            sizes[b] = sizes.get(b, 0) + 1
        if all(v == 1 for v in sizes.values()):
            continue
        # accepting status per block (not all same)
        block_acc = [randomizer.choice([True, False]) for _ in range(nblocks)]
        if all(block_acc) or not any(block_acc):
            # pick random but ensure mix
            block_acc[randomizer.randrange(nblocks)] = not block_acc[randomizer.randrange(nblocks)]
        # transitions per block: for each symbol, a block destination
        block_trans = [[randomizer.randrange(nblocks) for _ in symbols]
                       for _ in range(nblocks)]
        # build per-state transition by following its block's transitions
        trans = []
        for i in range(n):
            bi = block_assign[i]
            trans.append([_first_state_of_block(block_dst, block_assign)
                          for block_dst in block_trans[bi]])
        accepting = {i for i in range(n) if block_acc[block_assign[i]]}
        start = randomizer.randrange(n)
        # require both an accepting and a rejecting state reachable from start
        if not (_reach_accept(trans, start, n, accepting)
                and _reach_reject(trans, start, n, accepting)):
            continue
        # verify: the Moore refinement should produce exactly nblocks blocks,
        # no fewer and no more, guaranteeing the intended merges are exactly
        # the intended ones.
        labels = _minimize(list(range(n)), symbols, trans, accepting)
        block_of = _canonical_block(labels)
        num_blocks = len(set(block_of))
        if num_blocks != nblocks:
            continue
        if num_blocks == n or num_blocks < 2:
            continue
        tokens = [f"q{i}" for i in range(n)]
        return tokens, symbols, trans, accepting, start, block_of, num_blocks


def _first_state_of_block(block, assign):
    for i, b in enumerate(assign):
        if b == block:
            return i
    return 0


def _reach_accept(trans, start, n, accepting):
    seen = [False] * n
    stack = [start]
    seen[start] = True
    while stack:
        s = stack.pop()
        if s in accepting:
            return True
        for c in range(len(trans[s])):
            d = trans[s][c]
            if not seen[d]:
                seen[d] = True
                stack.append(d)
    return False


def _reach_reject(trans, start, n, accepting):
    seen = [False] * n
    stack = [start]
    seen[start] = True
    while stack:
        s = stack.pop()
        if s not in accepting:
            return True
        for c in range(len(trans[s])):
            d = trans[s][c]
            if not seen[d]:
                seen[d] = True
                stack.append(d)
    return False


def _sorted_block_list(block_of):
    """Return the partition as a sorted list of sorted blocks of state names."""
    n = len(block_of)
    blocks = {}
    for i in range(n):
        b = block_of[i]
        blocks.setdefault(b, []).append(f"q{i}")
    out = [",".join(sorted(lst)) for lst in blocks.values()]
    out.sort()
    return " | ".join(out)


class DfaMinimization(Task):
    summary = ("Compute Moore partition refinement over complete deterministic finite automata "
               "with varied state counts, alphabet sizes, and accepting sets, and emit the "
               "canonical partition of minimized states as a canonical merged list.")
    config_cls = DFAConfig

    def generate_entry(self):
        cfg = self.config
        alphabet = list(range(cfg.alphabet_size))
        for _ in range(200):
            tokens, symbols, trans, accepting, start, block_of, num_blocks = generate_one(
                cfg.max_states, cfg.min_states, alphabet, random)
            if num_blocks >= 2:
                break
        else:
            raise RuntimeError("failed to build non-trivial dfa")
        payload = {
            "states": tokens,
            "alphabet": symbols,
            "transitions": {tokens[i]: {symbols[c]: tokens[int(trans[i][c])] for c in range(len(symbols))}
                            for i in range(len(tokens))},
            "accepting": sorted((f"q{i}" for i in range(len(tokens)) if i in accepting)),
            "start": tokens[start],
        }
        answer = _sorted_block_list(block_of)
        metadata = edict({"payload": payload, "block_of": block_of})
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        p = metadata.payload
        trans_lines = "\n".join(f"from {s}: " + ", ".join(f"{c}->{d}" for c, d in t.items())
                                for s, t in sorted(p["transitions"].items()))
        acc = ", ".join(p["accepting"])
        prompt = (
            f"Consider the deterministic finite automaton (DFA) over alphabet "
            f"{{{', '.join(p['alphabet'])}}} with start state {p['start']}.\n"
            f"States: {', '.join(p['states'])}.\n"
            f"{trans_lines}\n"
            f"Accepting states: {acc}.\n"
            f"Use Moore's partition refinement to minimize this DFA, merging the "
            f"equivalent states. Give the canonical minimized partition as a list of "
            f"blocks, each block written as its member states sorted alphabetically "
            f"and joined by commas, blocks separated by ' | ', and the whole list "
            f"sorted lexicographically.\n\nThe answer is the partition string."
        )
        return prompt

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        got = answer.strip()
        if got == entry.answer:
            return 1.0
        # accept any of several harmless spacings
        norm = _normalize(got)
        gold = _normalize(entry.answer)
        return 1.0 if norm == gold else 0.0


def _normalize(s):
    return ",".join("".join(ch for ch in block if ch.isalnum() or ch == ",")
                    for block in s.split("|"))
