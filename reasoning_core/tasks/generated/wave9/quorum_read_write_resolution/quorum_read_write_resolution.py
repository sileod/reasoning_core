"""Quorum read/write resolution: evaluate a linearized sequence of quorum writes
and reads across replicas and report the value observed by a queried read.

A fixed set of replicas holds versioned values. Each write targets a quorum of
replicas (a minimum number that must acknowledge the new value and version).
Each read is issued to a quorum of replicas and returns the value of the newest
version observed among them, using a version tie-break when multiple replicas
differ. The task asks for the value visible to a specified read operation given
the full schedule of writes and reads and their quorum assignments.
"""

import random
import ast

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


def _pick_quorum(pop, k, rng):
    return sorted(rng.sample(pop, k))


def _apply_write(state, quorum, value):
    for r in quorum:
        state[r] = value  # value is a (version, value) pair from caller


def _run_read(state, quorum, version_clause):
    # newest version observed among quorum, returning (max_version, value_at_that_version)
    best_v = -1
    best_val = None
    for r in quorum:
        v, val = state[r]
        if v > best_v:
            best_v = v
            best_val = val
    return best_v, best_val


class QuorumConfig(Config):
    n_replicas: int = 4
    n_ops: int = 6
    min_quorum_frac: float = 0.5
    value_range: int = 5

    def apply_difficulty(self, level):
        self.n_replicas = sround(self.n_replicas + 1 + level // 2)
        self.n_ops = sround(self.n_ops + 2 * level)
        self.min_quorum_frac = min(0.5 + 0.08 * level, 0.85)
        self.value_range = sround(self.value_range + level)


def _score_answer(answer, entry):
    try:
        a = int(str(answer).strip())
        g = int(str(entry.answer).strip())
    except Exception:
        return 0.0
    return 1.0 if a == g else 0.0


class QuorumReadWriteResolution(Task):
    summary = ("Evaluate versioned quorum writes and reads across replicas with explicit "
               "quorum sizes, version tie-breaks and value assignments, returning the exact "
               "value visible to a queried read operation.")
    config_cls = QuorumConfig
    task_version = 2

    def generate_entry(self):
        c = self.config
        rng = random
        n = c.n_replicas
        replicas = list(range(n))
        values = list(range(c.value_range))

        # Decide how many ops are reads vs writes, with at least one queried read.
        n_reads = rng.randint(1, max(1, c.n_ops // 2))
        n_writes = max(1, c.n_ops - n_reads)
        n_ops = n_writes + n_reads

        min_q = max(1, int(n * c.min_quorum_frac))
        max_q = n

        history = []          # list of (optype, quorum, value) ops; reads have value None
        read_ids = []         # indices into history that are reads
        state = {r: (-1, -1) for r in replicas}  # (version, value); version -1 = unset
        version = 0

        for _ in range(n_writes):
            wq_size = rng.randint(min_q, max_q)
            wq = _pick_quorum(replicas, wq_size, rng)
            value = rng.choice(values)
            version += 1
            history.append(("write", wq, value))
            _apply_write(state, wq, (version, value))

        for _ in range(n_reads):
            rq_size = rng.randint(min_q, max_q)
            rq = _pick_quorum(replicas, rq_size, rng)
            history.append(("read", rq, None))

        # read_ids covers all reads after the point we care; choose a queried read that
        # happens after at least one write so there is a well-defined visible value.
        read_indices = [i for i, (t, _, _) in enumerate(history) if t == "read"]
        queried = rng.choice(read_indices)

        # Recompute the actual observable value for the queried read using fresh state.
        st = {r: (-1, -1) for r in replicas}
        vnum = 0
        for i, (t, q, v) in enumerate(history):
            if t == "write":
                vnum += 1
                _apply_write(st, q, (vnum, v))
            else:
                if i == queried:
                    best_v, best_val = _run_read(st, q, None)
                    if best_val == -1:
                        # no replica holds any value yet; reject and resample
                        raise RuntimeError("queried read before any value")
                    answer = int(best_val)

        # Verifier: independent replay must reproduce the answer.
        st2 = {r: (-1, -1) for r in replicas}
        rq = history[queried][1]
        vnum2 = 0
        for i, (t, q, v) in enumerate(history):
            if i >= queried:
                break
            if t == "write":
                vnum2 += 1
                _apply_write(st2, q, (vnum2, v))
        _, val2 = _run_read(st2, rq, None)
        assert int(val2) == answer, (val2, answer)

        ops_repr = []
        for t, q, v in history:
            if t == "write":
                ops_repr.append("write(%s) = %d" % (",".join(str(x) for x in q), v))
            else:
                ops_repr.append("read(%s)" % (",".join(str(x) for x in q)))

        metadata = edict({
            "n_replicas": n,
            "min_q": min_q,
            "max_q": max_q,
            "ops": ops_repr,
            "queried_read": queried,
        })
        metadata.payload = {
            "n_replicas": n,
            "ops": ops_repr,
            "queried_read": queried,
        }
        return Entry(metadata=metadata, answer=str(answer))

    def render_prompt(self, metadata):
        lines = [
            "There are %d replicas, each storing a value with a version. Values are "
            "non-negative integers; a higher version is newer. Initially every replica "
            "is unset (holds no value)." % metadata.n_replicas,
            "Operations happen in order. A write(V) = x writes value x at a new version "
            "(each write's version is one higher than the previous write) to every replica "
            "in the listed set. A read(S) samples the replicas in set S and returns the "
            "value of the newest version among them; if several replicas hold that same "
            "newest version they all agree on its value by construction.",
            "Schedule:",
        ]
        for idx, op in enumerate(metadata.ops):
            lines.append("%d: %s" % (idx, op))
        lines.append("")
        lines.append("Question: list the values visible to every read operation, then give "
                     "the value seen by the read at index %d as the answer." % metadata.queried_read)
        lines.append("The answer is a single non-negative integer.")
        return render_payload(metadata.payload) + "\n" + "\n".join(lines)


TASK_META = {'parent_source_id': None,
 'idea': 'quorum_read_write_resolution (draw 1 of 1)',
 'hypothesis': 'HV-079',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/quorum_read_write_resolution',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1175081579,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
