import random
from collections import deque
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict

TASK_META = {'parent_source_id': None,
 'idea': 'transaction_serializability (draw 1 of 1)',
 'hypothesis': 'HV-049',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/transaction_serializability',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3323899505,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _precedence_edges(history, ntr):
    """history: list of (txn_idx, op, item). Return precedence edges (i,j)."""
    edges = []
    key = {}
    n_hist = len(history)
    for i in range(n_hist):
        idx, op, item = history[i]
        for j in range(i + 1, n_hist):
            jdx, jop, jitem = history[j]
            if idx == jdx:
                continue
            conflict = False
            if op == 'w' and jop in ('r', 'w') and item == jitem:
                conflict = True
            elif op == 'r' and jop == 'w' and item == jitem:
                conflict = True
            if conflict:
                e = (idx, jdx)
                if e not in key:
                    key[e] = True
                    edges.append(e)
    return edges


def _topo(edges, ntr):
    indeg = [0] * ntr
    adj = [[] for _ in range(ntr)]
    for a, b in edges:
        adj[a].append(b)
        indeg[b] += 1
    zero = sorted(i for i in range(ntr) if indeg[i] == 0)
    order = []
    while zero:
        u = zero.pop(0)
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                zero.append(v)
                zero.sort()
    if len(order) != ntr:
        return None
    return order


def _cycle_core(edges, ntr):
    """Transaction ids that lie on at least one directed precedence cycle."""
    adj = [set() for _ in range(ntr)]
    indeg = [0] * ntr
    outdeg = [0] * ntr
    for a, b in edges:
        adj[a].add(b)
        indeg[b] += 1
        outdeg[a] += 1
    alive = [True] * ntr
    dq = deque(i for i in range(ntr) if indeg[i] == 0)
    while dq:
        u = dq.popleft()
        if not alive[u]:
            continue
        alive[u] = False
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                dq.append(v)
    dq2 = deque(i for i in range(ntr) if alive[i] and outdeg[i] == 0)
    while dq2:
        u = dq2.popleft()
        if not alive[u]:
            continue
        alive[u] = False
        for prev in range(ntr):
            if alive[prev] and u in adj[prev]:
                outdeg[prev] -= 1
                if outdeg[prev] == 0:
                    dq2.append(prev)
    return [i for i in range(ntr) if alive[i]]


def _serialize(history, ntr):
    return _topo(_precedence_edges(history, ntr), ntr)


def _random_interleave(seq, ntr):
    queues = {t: list(v) for t, v in seq.items()}
    avail = [t for t in range(ntr) if queues[t]]
    out = []
    total = sum(len(v) for v in seq.values())
    for _ in range(total):
        t = random.choice(avail)
        op, item = queues[t].pop(0)
        out.append((t, op, item))
        if not queues[t]:
            avail.remove(t)
    return out


@dataclass
class SnapshotTxnConfig(Config):
    n_txns: int = 3
    n_ops: int = 6
    n_items: int = 3

    def apply_difficulty(self, level):
        self.n_txns = 3 + level
        self.n_ops = 6 + 3 * level
        self.n_items = 3 + level


class Serializability(Task):
    summary = ("Build conflict-precedence relations from interleaved transaction "
               "reads and writes, returning serializability and a canonical serial "
               "order when one exists.")
    config_cls = SnapshotTxnConfig

    def generate_entry(self):
        cfg = self.config
        ntr = cfg.n_txns
        n_items = cfg.n_items
        max_tries = 800

        for _ in range(max_tries):
            serializable = random.random() < 0.5
            seq = {}
            for _ in range(cfg.n_ops):
                t = random.randrange(ntr)
                op = random.choice(['r', 'w'])
                item = random.randrange(n_items)
                seq.setdefault(t, []).append((op, item))
            for t in range(ntr):
                seq.setdefault(t, [('r', random.randrange(n_items))])

            if serializable:
                hist = _random_interleave(seq, ntr)
                order = _serialize(hist, ntr)
                if order is not None:
                    answer = "serial " + ",".join("T{}".format(i) for i in order)
                    break
            else:
                hist = _random_interleave(seq, ntr)
                core = _cycle_core(_precedence_edges(hist, ntr), ntr)
                if core:
                    answer = "nonserial " + ",".join("T{}".format(i) for i in sorted(core))
                    break
        else:
            raise RuntimeError("unable to generate an instance")

        if answer.startswith("serial"):
            assert _serialize(hist, ntr) is not None
        else:
            assert _serialize(hist, ntr) is None

        history_render = ["T{}{}{}".format(t, op, item) for (t, op, item) in hist]
        metadata = edict({
            "history": history_render,
            "ntr": ntr,
        })
        metadata.payload = {"history": history_render, "ntr": ntr}
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = ["{}. {}".format(i + 1, h) for i, h in enumerate(metadata.payload["history"])]
        hist = "\n".join(lines)
        return (
            "Consider a database schedule of transactions. Each line lists one "
            "operation 'T<id><r|w><item>': transaction T<id> reads or writes a "
            "single item, in the order the lines appear. Two operations conflict "
            "if they act on the same item and at least one is a write. Build the "
            "conflict-precedence graph: for a write by Ti then a read or write by "
            "Tj on the same item, and for a read by Ti then a write by Tj on the "
            "same item, put edge Ti -> Tj.\n\n"
            "{hist}\n\n"
            "The schedule is conflict-serializable exactly when this graph has no "
            "directed cycle, and then a serial order is any topological ordering "
            "of the graph. Decide whether the schedule is conflict-serializable. "
            "If it is, give the answer as 'serial T0,T2,T1' with the transaction "
            "ids in the lexicographically first topological order, comma-separated. "
            "If it is not serializable, give the answer as 'nonserial T1,T3' "
            "listing (in sorted order) every transaction id that lies on at least "
            "one directed precedence cycle."
        ).format(hist=hist)

    def score_answer(self, answer, entry):
        gold = entry.answer
        a = answer.strip()
        history = _parse_history(entry.metadata["history"])
        ntr = entry.metadata["ntr"]

        if gold.startswith("serial"):
            if a == gold:
                return 1.0
            try:
                if not a.startswith("serial "):
                    return 0.0
                seq = [int(s.strip().lstrip("T")) for s in a[len("serial "):].split(",")]
            except Exception:
                return 0.0
            if sorted(seq) != list(range(ntr)):
                return 0.0
            order = _serialize(history, ntr)
            if order is None:
                return 0.0
            # verify it's a valid topological order respecting precedence
            pos = {v: i for i, v in enumerate(seq)}
            for e, f in _precedence_edges(history, ntr):
                if pos[e] > pos[f]:
                    return 0.0
            return 1.0
        else:
            if a == gold:
                return 1.0
            try:
                if not a.startswith("nonserial "):
                    return 0.0
                listed = sorted(int(s.strip().lstrip("T")) for s in a[len("nonserial "):].split(","))
            except Exception:
                return 0.0
            core = _cycle_core(_precedence_edges(history, ntr), ntr)
            if sorted(core) != listed:
                return 0.0
            if _serialize(history, ntr) is not None:
                return 0.0
            return 1.0


def _parse_history(history_render):
    out = []
    for h in history_render:
        t = int(h[1])
        op = h[2]
        item = int(h[3:])
        out.append((t, op, item))
    return out
