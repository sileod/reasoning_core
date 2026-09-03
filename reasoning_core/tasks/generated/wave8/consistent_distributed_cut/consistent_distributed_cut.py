import random
from dataclasses import dataclass

from reasoning_core.template import Entry, Config, Task, edict, stochastic_rounding as sround


@dataclass
class ConsistentDistributedCutConfig(Config):
    n_procs: int = 3
    size: int = 6
    n_msgs: int = 10
    orphan_prob: float = 0.5
    max_attempts: int = 50

    def apply_difficulty(self, level):
        self.n_procs = sround(2 + level)
        self.size = sround(4 + level)
        self.n_msgs = sround(8 + 3 * level)
        self.max_attempts = sround(50 + 20 * level)


def _count_orphans(messages, cuts):
    n = 0
    for u, ps, v, pr in messages:
        if ps <= cuts[u] and pr > cuts[v]:
            n += 1
    return n


def _sample_message(u, v, cu, cv, su, sv, want_orphan):
    if want_orphan and cu >= 1 and cv < sv:
        ps = random.randint(1, cu)
        pr = random.randint(cv + 1, sv)
        return ps, pr
    opts = []
    if cu < su:
        opts.append("exclude_send")
    if cv >= 1:
        opts.append("include_recv")
    if not opts:
        ps = 1
        pr = sv
        return ps, pr
    mode = random.choice(opts)
    if mode == "exclude_send":
        ps = random.randint(cu + 1, su)
        pr = random.randint(1, sv)
    else:
        ps = random.randint(1, su)
        pr = random.randint(1, cv)
    return ps, pr


class ConsistentDistributedCut(Task):
    summary = "Given per-process sizes, a selected global cut, and message send/receive edges, report the number of orphaned messages that cross the cut inconsistently."
    config_cls = ConsistentDistributedCutConfig

    def generate_entry(self):
        cfg = self.config
        for _ in range(cfg.max_attempts):
            sizes = [max(3, cfg.size + random.randint(-1, 1)) for _ in range(cfg.n_procs)]
            cuts = [random.randint(0, s) for s in sizes]
            messages = []
            want = [random.random() < cfg.orphan_prob for _ in range(cfg.n_msgs)]
            for m in range(cfg.n_msgs):
                u = random.randrange(cfg.n_procs)
                v = random.randrange(cfg.n_procs - 1)
                if v >= u:
                    v += 1
                cu, cv = cuts[u], cuts[v]
                ps, pr = _sample_message(u, v, cu, cv, sizes[u], sizes[v], want[m])
                messages.append((u, ps, v, pr))
            count = _count_orphans(messages, cuts)
            if not (0 <= count <= cfg.n_msgs):
                continue
            metadata = edict(
                sizes=[int(s) for s in sizes],
                cuts=[int(c) for c in cuts],
                messages=[(int(a), int(b), int(c), int(d)) for a, b, c, d in messages],
            )
            metadata.payload = {"process sizes": metadata.sizes, "cut points": metadata.cuts, "message edges": metadata.messages}
            return Entry(metadata=metadata, answer=str(count))
        raise RuntimeError("Failed to generate a consistent-cut instance")

    def render_prompt(self, metadata):
        proc = " ".join(f"P{p}:{metadata.sizes[p]}" for p in range(len(metadata.sizes)))
        cut = " ".join(f"P{p}:{metadata.cuts[p]}" for p in range(len(metadata.cuts)))
        edges = "; ".join(
            f"m{m} P{u}@{ps}->P{v}@{pr}" for m, (u, ps, v, pr) in enumerate(metadata.messages)
        )
        return (
            f"In a message-passing system, process P0..P{len(metadata.sizes) - 1} each execute an ordered "
            f"sequence of events; process sizes (number of events): {proc}.\n"
            f"A global cut selects a prefix of each process: {cut} (cut point = number of included events, "
            "can be 0 up to the process size).\n"
            f"A message is sent by one process (at its send position, 1=first event) and received by another "
            f"(at its receive position). Message edges: {edges}.\n"
            "A message is orphaned (the cut is inconsistent at that message) when its send position lies at or "
            "before the sender's cut point but its receive position lies strictly after the receiver's cut point.\n"
            "How many messages are orphaned by this cut? The answer is a non-negative integer: the number of "
            "orphaned messages."
        )

    def score_answer(self, answer, entry):
        try:
            val = int(str(answer).split("=")[-1].strip().rstrip("."))
        except (ValueError, TypeError):
            return 0.0
        ref = int(entry.answer)
        return float(val == ref)


TASK_META = {'parent_source_id': None,
 'idea': 'consistent_distributed_cut (draw 1 of 2)',
 'hypothesis': 'W1-037',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/consistent_distributed_cut',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3066933245,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
