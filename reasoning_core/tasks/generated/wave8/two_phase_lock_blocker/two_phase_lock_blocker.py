"""Strict 2PL lock blocker identification.

Given a sequence of lock requests with unlock events interleaved under strict
two-phase locking, name the transaction whose outstanding locks block the most
recent request, or None if no transaction blocks it.
"""

import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload


TASK_META = {'parent_source_id': None,
 'idea': 'two_phase_lock_blocker (draw 1 of 2)',
 'hypothesis': 'W1-034',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/two_phase_lock_blocker',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 236351923,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class TwoPhaseLockBlockerConfig(Config):
    n_txns: int = 3
    n_items: int = 5
    depth: int = 5

    def apply_difficulty(self, level):
        self.n_txns = sround_bounded(3 + level)
        self.n_items = sround_bounded(5 + 2 * level)
        self.depth = sround_bounded(6 + 4 * level)


def sround_bounded(v):
    return int(v)


def compute_answer(sequence, txns, items):
    """Return the txn id blocking the final request, or None.

    sequence: list of (txn, item, 'L' or 'U') events.
    Returns the set's single blocking txn name or None.
    """
    held = {t: set() for t in txns}
    for i, (t, item, op) in enumerate(sequence):
        is_last = (i == len(sequence) - 1)
        if op == 'L':
            blockers = sorted(
                name for name in txns
                if name != t and item in held[name]
            )
            if is_last:
                return blockers[0] if blockers else None
            if blockers:
                continue
            held[t].add(item)
        else:
            held[t].discard(item)
    return None


def generate_sequence(level):
    txns = ['T%d' % i for i in range(1, 4 + level)]
    items = ['D%d' % i for i in range(1, 6 + 2 * level)]
    depth = 6 + 4 * level
    held = {t: set() for t in txns}
    sequence = []
    bounded = 0
    while len(sequence) < depth and bounded < depth * 4:
        bounded += 1
        if random.random() < 0.6:
            t = random.choice(txns)
            item = random.choice(items)
            blockers = [n for n in txns if n != t and item in held[n]]
            if not blockers:
                held[t].add(item)
            sequence.append((t, item, 'L'))
        else:
            t = random.choice(txns)
            if held[t]:
                item = random.choice(sorted(held[t]))
                held[t].discard(item)
                sequence.append((t, item, 'U'))
    # final request: roughly balance blocked vs unblocked
    t = random.choice(txns)
    item = random.choice(items)
    blockers = sorted(n for n in txns if n != t and item in held[n])
    if random.random() < 0.7:
        if not blockers:
            # force a blocker: pick an item held by a different txn
            candidates = [n for n in txns if n != t and held[n]]
            if candidates:
                b = random.choice(candidates)
                item = random.choice(sorted(held[b]))
                blockers = sorted(n for n in txns if n != t and item in held[n])
    sequence.append((t, item, 'L'))
    return sequence, txns, items, blockers


class TwoPhaseLockBlocker(Task):
    summary = "Given lock requests under strict 2PL, output the transaction blocking a queried request or None."
    config_cls = TwoPhaseLockBlockerConfig

    def generate_entry(self):
        level = self.config.level
        sequence, txns, items, blockers = generate_sequence(level)
        answer = compute_answer(sequence, txns, items)
        if answer is None:
            answer_str = "None"
        else:
            answer_str = answer
        # ensure a mix of None and txn answers
        metadata = edict({
            "sequence": sequence,
            "txns": txns,
            "items": items,
            "final": sequence[-1],
            "blockers": blockers,
        })
        metadata.payload = {
            "sequence": [[t, it, op] for (t, it, op) in sequence],
        }
        return Entry(metadata=metadata, answer=answer_str)

    def render_prompt(self, metadata):
        ops = []
        for (t, it, op) in metadata.sequence:
            if op == 'L':
                ops.append("%s requests lock on %s" % (t, it))
            else:
                ops.append("%s releases lock on %s" % (t, it))
        body = "\n".join(ops)
        return (
            "Transactions follow strict two-phase locking: a transaction may request "
            "any number of locks but releases them only after all its requests, and "
            "a lock request is granted only if no other transaction currently holds it.\n\n"
            + body + "\n\n"
            "The final request is the queried request. Name the transaction whose held "
            "lock blocks that request; if no transaction blocks it, the answer "
            "is the token None.\n"
            "The answer is a single transaction id such as T1, or the token None when "
            "no transaction blocks the queried request."
        )

    def score_answer(self, answer, entry):
        gold = entry.answer
        a = answer.strip()
        g = gold.strip()
        if a == g:
            return 1.0
        return 0.0
