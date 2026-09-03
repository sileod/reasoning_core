"""Conflict serializability: decide conflict-serializability of a transaction
schedule and, when serial, report the unique equivalent serial order.

A schedule is a sequence of operations rT(A) / wT(A) (transaction read/write of
an item). Two operations (from different transactions) are in conflict when they
touch the same item and at least one is a write. A schedule is conflict
serializable iff its precedence (conflict) graph over transactions is acyclic;
the unique conflict-equivalent serial order is then the (unique) topological
order of that graph. When the conflict graph has more than one topological
order the schedule has several equivalent serial orders, so the answer is
"NO" (not conflict serializable) only when the graph has a cycle, and a
"YES" answer is accompanied by the unique serial order.

Because the generator always builds a schedule whose conflict graph has exactly
one topological order whenever it is acyclic, the canonical answer is either
"YES:<order>" or "NO".  We independently verify with networkx.
"""

import random
from dataclasses import dataclass

import networkx as nx

from reasoning_core.template import Task, Entry, Config, edict, render_payload


def _serial_order(precedence_edges, n_trans):
    """Return unique topological order over 1..n_trans or None if cyclic/not unique."""
    g = nx.DiGraph()
    g.add_nodes_from(range(1, n_trans + 1))
    g.add_edges_from(precedence_edges)
    try:
        order = list(nx.topological_sort(g))
    except nx.NetworkXUnfeasible:
        return None
    it = nx.all_topological_sorts(g)
    try:
        first = next(it)
    except StopIteration:
        return None
    try:
        next(it)
        return None
    except StopIteration:
        return tuple(first)


def _make_serializable(config):
    """Build a schedule whose conflict graph is a DAG with a unique topo order.

    Strategy: choose a random total order of transactions (the serial order),
    then assign operations to items so that the resulting precedence graph has
    that order as its UNIQUE topological order. We verify uniqueness with
    networkx and resample the schedule until it holds.
    """
    n_items = random.randint(config.n_items_min, config.n_items_max)
    n_trans = random.randint(config.n_trans_min, config.n_trans_max)

    for _ in range(3000):
        # pick a random target serial order (a random permutation)
        order = list(range(1, n_trans + 1))
        random.shuffle(order)

        # each transaction gets 1..3 operations on random items
        transaction_ops = {t: [] for t in range(1, n_trans + 1)}
        for t in order:
            for _ in range(random.randint(1, 3)):
                item = random.randint(1, n_items)
                transaction_ops[t].append((random.choice(("r", "w")), item))

        # gather per-item operation lists preserving schedule = concatenation
        # of transactions in the chosen serial order
        item_ops = {i: [] for i in range(1, n_items + 1)}
        seq = []
        for t in order:
            for (w, item) in transaction_ops[t]:
                item_ops[item].append((w, t))
                seq.append((w, t, item))

        # augment: ensure each consecutive pair (order[k], order[k+1]) shares a
        # conflicting item so the chain edges exist (kills alternative orders)
        ok = True
        for k in range(n_trans - 1):
            ta, tb = order[k], order[k + 1]
            item = random.randint(1, n_items)
            item_ops[item].append(("w", ta))
            item_ops[item].append(("r", tb))
            seq.append(("w", ta, item))
            seq.append(("r", tb, item))

        precedence = set()
        for item, lst in item_ops.items():
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    wa, ta = lst[a]
                    wb, tb = lst[b]
                    if ta == tb:
                        continue
                    if wa == "w" or wb == "w":
                        precedence.add((ta, tb))

        uniq = _serial_order(precedence, n_trans)
        if uniq is None:
            continue  # not a unique-order DAG, resample
        random.shuffle(seq)
        return seq, precedence, n_trans, uniq
    return None


def _make_non_serializable(n_trans, n_items):
    """Build a schedule whose conflict graph is cyclic (-> not serializable).

    Returns answer "NO" whenever the schedule is not conflict serializable.
    """
    for _ in range(2000):
        ops = []
        item_ops = {i: [] for i in range(1, n_items + 1)}
        for _ in range(random.randint(2, 6)):
            t = random.randint(1, n_trans)
            i = random.randint(1, n_items)
            w = random.choice(("r", "w"))
            ops.append((w, t, i))
            item_ops[i].append((w, t))

        precedence = set()
        for item, lst in item_ops.items():
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    wa, ta = lst[a]
                    wb, tb = lst[b]
                    if ta == tb:
                        continue
                    if wa == "w" or wb == "w":
                        precedence.add((ta, tb))

        order = _serial_order(precedence, n_trans)
        if order is None and precedence:
            # cyclic graph => not conflict serializable
            return ops, precedence, n_trans, None
    return None


def _render_schedule(ops):
    parts = []
    for (w, t, item) in ops:
        parts.append(f"{w}{t}({item})")
    return ",".join(parts)


@dataclass
class ConflictSerializabilityConfig(Config):
    n_items_min: int = 2
    n_items_max: int = 3
    n_trans_min: int = 3
    n_trans_max: int = 4

    def apply_difficulty(self, level):
        self.n_items_min = 2 + level
        self.n_items_max = 4 + level
        self.n_trans_min = 3 + level
        self.n_trans_max = 5 + level


class ConflictSerializability(Task):
    summary = ("Given a transaction schedule, determine conflict serializability and give the "
               "unique serial order if one exists, answering YES:<order> or NO.")
    config_cls = ConflictSerializabilityConfig

    def generate_entry(self):
        cfg = self.config
        if random.random() < 0.78:
            built = None
            for _ in range(2000):
                built = _make_serializable(cfg)
                if built is not None:
                    break
            if built is None:
                raise RuntimeError("could not build serializable instance")
            ops, precedence, n_trans, order = built
            serial = True
            answer = "YES:" + ",".join(str(t) for t in order)
        else:
            built = None
            for _ in range(2000):
                built = _make_non_serializable(n_trans_rng(), cfg.n_items_max)
                if built is not None:
                    break
            if built is None:
                raise RuntimeError("could not build non-serializable instance")
            ops, precedence, n_trans, order = built
            serial = False
            answer = "NO"

        rendered = _render_schedule(ops)
        payload = {"schedule": rendered, "answer_format_hint": "YES:T1,T2,... or NO"}

        # independent verification
        ver = _serial_order(precedence, n_trans)
        if serial:
            assert ver is not None, "claimed serial but graph not acyclic-with-unique-order"
            assert ",".join(str(t) for t in ver) == answer.split(":", 1)[1]
        else:
            assert ver is None, "claimed NO but found a valid serial order"

        metadata = edict({
            "schedule": rendered,
            "n_trans": int(n_trans),
            "n_items": int(max(item for (_, _, item) in ops)),
            "precedence_edges": sorted((int(a), int(b)) for (a, b) in precedence),
        })
        metadata.payload = payload
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (
            f"Consider the following transaction schedule. Operations are "
            f"rT(A) (transaction T reads item A) and wT(A) (transaction T writes item A). "
            f"Two operations from different transactions conflict if they touch the same item "
            f"and at least one is a write. A schedule is conflict-serializable iff the "
            f"precedence graph (edge Ti->Tj when an operation of Ti precedes a conflicting "
            f"operation of Tj) is acyclic, and its unique topological order is the conflict-"
            f"equivalent serial order.\n\n"
            f"Schedule: {metadata.schedule}\n\n"
            f"Decide whether the schedule is conflict-serializable. If YES, give the unique "
            f"serial order of transactions. The answer format is exactly 'YES:T1,T2,...' (the "
            f"serial order, a full permutation of all transactions) if it is conflict-"
            f"serializable, or exactly 'NO' if it is not.\n"
            f"Example: 'YES:2,1,3' or 'NO'. The answer is a single exact string."
        )

    def score_answer(self, answer, entry):
        gold = entry.answer
        if not isinstance(answer, str):
            return 0.0
        a = answer.strip()
        if a == gold:
            return 1.0
        return 0.0


def n_trans_rng():
    return random.randint(2, 5)


TASK_META = {'parent_source_id': None,
 'idea': 'conflict_serializability (draw 1 of 2)',
 'hypothesis': 'W1-031',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/conflict_serializability',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2300412284,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
