import random

from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround


def _topo_order(n, succ):
    indeg = [0] * n
    for u in range(n):
        for v in succ[u]:
            indeg[v] += 1
    from collections import deque
    q = deque([u for u in range(n) if indeg[u] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != n:
        return None
    return order


def _reverse_order(n, succ):
    topo = _topo_order(n, succ)
    if topo is None:
        return None
    return list(reversed(topo))


def _live_analysis(n, succ, gen, kill, rev_order):
    live_in = [set() for _ in range(n)]
    live_out = [set() for _ in range(n)]
    changed = True
    while changed:
        changed = False
        for b in rev_order:
            new_out = set()
            for s in succ[b]:
                new_out |= live_in[s]
            if new_out != live_out[b]:
                live_out[b] = new_out
                changed = True
            new_in = (live_out[b] - kill[b]) | gen[b]
            if new_in != live_in[b]:
                live_in[b] = new_in
                changed = True
    return live_in, live_out


def _var_names(n_vars):
    return [chr(ord('a') + i) for i in range(n_vars)]


def _format_set(s, var_names):
    if not s:
        return "{}"
    return "{" + ",".join(var_names[i] for i in sorted(s)) + "}"


class CommonConfig(Config):
    n_blocks: int = 5
    n_vars: int = 4
    edge_prob: float = 0.5

    def apply_difficulty(self, level):
        self.n_blocks = sround(self.n_blocks + level)
        self.n_vars = sround(self.n_vars + level)
        self.edge_prob = min(0.9, 0.4 + 0.08 * level)


class DataflowLiveness(Task):
    summary = "Compute live-in and live-out variable sets over control-flow graphs using standard backward dataflow equations and a stated iteration schedule."
    config_cls = CommonConfig

    def generate_entry(self):
        c = self.config
        n = c.n_blocks
        n_vars = c.n_vars
        var_names = _var_names(n_vars)
        max_attempts = 200
        for _ in range(max_attempts):
            succ = [[] for _ in range(n)]
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < c.edge_prob:
                        succ[u].append(v)
            for u in range(n - 1):
                if not succ[u]:
                    succ[u].append(u + 1)
            rev_order = _reverse_order(n, succ)
            if rev_order is None:
                continue
            gen = []
            kill = []
            for _ in range(n):
                num_gen = random.randint(1, n_vars)
                gs = random.sample(range(n_vars), num_gen)
                num_kill = random.randint(1, n_vars)
                ks = random.sample(range(n_vars), num_kill)
                gen.append(set(gs))
                kill.append(set(ks))
            for b in range(n):
                for v in gen[b]:
                    if v in kill[b]:
                        kill[b].discard(v)
            live_in, live_out = _live_analysis(n, succ, gen, kill, rev_order)
            live_in_str = [_format_set(s, var_names) for s in live_in]
            live_out_str = [_format_set(s, var_names) for s in live_out]
            succ_str = []
            for u in range(n):
                succ_str.append("[" + ",".join(str(v) for v in succ[u]) + "]")
            gen_str = [_format_set(s, var_names) for s in gen]
            kill_str = [_format_set(s, var_names) for s in kill]
            metadata = edict({
                "n_blocks": n,
                "n_vars": n_vars,
                "succ": succ_str,
                "gen": gen_str,
                "kill": kill_str,
                "order": [str(b) for b in rev_order],
                "live_in": live_in_str,
                "live_out": live_out_str,
                "answer": " ".join(live_in_str),
            })
            metadata.payload = {
                "succ": succ_str,
                "gen": gen_str,
                "kill": kill_str,
                "order": [str(b) for b in rev_order],
            }
            met = metadata.live_in
            counts = {}
            for s in met:
                counts[s] = counts.get(s, 0) + 1
            if len(counts) < 2:
                continue
            return Entry(metadata=metadata, answer=metadata.answer)
        raise RuntimeError("failed to build live example")

    def render_prompt(self, metadata):
        lines = []
        lines.append("A control-flow graph has blocks indexed 0..%d. The successor (outgoing edge)"
                     " lists are:" % (metadata.n_blocks - 1))
        lines.append("successors = " + ", ".join("B%d->%s" % (i, metadata.payload["succ"][i])
                                                for i in range(metadata.n_blocks)))
        lines.append("For each block, Gen is the set of variables defined (made live) there and Kill"
                     " is the set of variables killed (not live on exit):")
        for i in range(metadata.n_blocks):
            lines.append("B%d: Gen=%s Kill=%s" % (i, metadata.payload["gen"][i],
                                                  metadata.payload["kill"][i]))
        order = ",".join(str(int(b)) for b in metadata.payload["order"])
        lines.append("Compute live-in and live-out variable sets using the backward dataflow"
                     " equations LiveOut[b] = union of LiveIn of successors, LiveIn[b] ="
                     " (LiveOut[b] - Kill[b]) | Gen[b], iterated over blocks in reverse post-order"
                     " order %s until fixpoint." % order)
        lines.append("Give the answer as the live-in sets of blocks 0..%d in order, space-separated,"
                     " each set written as comma-separated variables inside braces, e.g. for 2"
                     " blocks: {a} {b,c}." % (metadata.n_blocks - 1))
        lines.append("The live-in sets are:")
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        gold = entry.answer
        if answer is None:
            return 0.0
        answer = "".join(answer.split())
        gold = "".join(gold.split())
        return 1.0 if answer == gold else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'dataflow_liveness (draw 1 of 1)',
 'hypothesis': 'HV-027',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/dataflow_liveness',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1583885757,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
