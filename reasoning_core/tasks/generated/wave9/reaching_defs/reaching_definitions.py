import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'reaching_definitions (draw 1 of 1)',
 'hypothesis': 'HV-028',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/reaching_definitions',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2905774178,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _fresh():
    return random.randint(100, 999)


class _CFG:
    """n nodes 0..n-1, node 0 entry, node n-1 exit. Edges (u,v)."""

    def __init__(self, edges):
        self.edges = edges
        self.n = max(max(e) for e in edges) + 1
        self.succ = [[] for _ in range(self.n)]
        self.pred = [[] for _ in range(self.n)]
        for (u, v) in edges:
            self.succ[u].append(v)
            self.pred[v].append(u)

    def make_defs(self, nvars):
        gen = []
        kill = []
        for i in range(self.n):
            g = _fresh()
            gen.append((g, i))
            prior = [gen[j][0] for j in range(i)]
            random.shuffle(prior)
            nk = random.randint(0, len(prior))
            kill.append(set(prior[:nk]))
        return gen, kill

    def reaching(self, entry, gen, kill):
        n = self.n
        out_set = [set() for _ in range(n)]
        in_set = [set() for _ in range(n)]
        out_set[entry] = {gen[entry][0]}
        order = self._rev_bfs_order(entry)
        changed = True
        while changed:
            changed = False
            for v in order:
                ins = set()
                for p in self.pred[v]:
                    if out_set[p]:
                        ins |= out_set[p]
                in_set[v] = ins
                newout = (ins - kill[v]) | {gen[v][0]}
                if newout != out_set[v]:
                    out_set[v] = newout
                    changed = True
        return in_set, out_set

    def _rev_bfs_order(self, entry):
        order = []
        stack = [entry]
        seen = set()
        while stack:
            v = stack.pop(0)
            if v in seen:
                continue
            seen.add(v)
            order.append(v)
            for w in self.succ[v]:
                if w not in seen:
                    stack.append(w)
        return order


def _build_cfg(n, n_branches, n_loops):
    edges = []
    nodes = list(range(n))
    for i in range(n - 1):
        edges.append((i, i + 1))
    branch_srcs = set()
    if n_branches > 0:
        last = n - 2
        pool = list(range(1, last))
        random.shuffle(pool)
        use = group_reject(pool, n_branches)
        for s in use:
            add_branch(edges, s, n)
    loop_srcs = set()
    if n_loops > 0:
        last = n - 2
        pool = list(range(1, last))
        random.shuffle(pool)
        use = group_reject(pool, n_loops)
        for s in use:
            add_loop(edges, s, n)
    return edges


def group_reject(pool, k):
    res = []
    for x in pool:
        if len(res) == k:
            break
        if all(abs(x - r) > 1 for r in res):
            res.append(x)
    return res


def add_branch(edges, s, n):
    if s + 1 < n:
        edges.append((s, s + 1))


def add_loop(edges, s, n):
    if s + 1 < n:
        edges.append((s + 1, s))


def _reachable(edges, n):
    succ = [[] for _ in range(n)]
    for (u, v) in edges:
        succ[u].append(v)
    seen = set()
    stack = [0]
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        stack.extend(succ[v])
    return seen


class ReachingDefinitionsConfig(Config):
    n_stmts: int = 6
    n_branches: int = 1
    n_loops: int = 0

    def apply_difficulty(self, level):
        self.n_stmts = sround(3 + level * 2)
        self.n_branches = sround(level)
        self.n_loops = sround(level // 2)


class ReachingDefs(Task):
    summary = ("Propagate reaching-definition sets through branched control flow and loops, "
               "returning the assignments that can reach a queried program point.")
    config_cls = ReachingDefinitionsConfig
    task_version = 2

    def generate_entry(self):
        c = self.config
        for _ in range(200):
            edges = _build_cfg(c.n_stmts, c.n_branches, c.n_loops)
            if len(_reachable(edges, c.n_stmts)) == c.n_stmts:
                break
        else:
            raise RuntimeError("could not build reachable cfg")
        cfg = _CFG(edges)
        gen, kill = cfg.make_defs(c.n_stmts)
        query = random.randint(0, cfg.n - 1)
        in_set, out_set = cfg.reaching(0, gen, kill)
        gold = sorted(in_set[query])
        answer = " ".join(str(x) for x in gold) if gold else "-"
        metadata = edict({
            "cfg": [list(e) for e in edges],
            "gen": [g for (g, _) in gen],
            "kill": [sorted(k) for k in kill],
            "query": query,
            "reaching_in": [sorted(list(s)) for s in in_set],
            "reaching_out": [sorted(list(s)) for s in out_set],
        })
        metadata.payload = {"cfg": metadata.cfg, "gen": metadata.gen, "kill": metadata.kill, "query": query}
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = ["%s -> %s" % (u, v) for (u, v) in metadata.cfg]
        defs_txt = []
        for i in range(len(metadata.gen)):
            kills = " ".join(str(x) for x in metadata.kill[i]) if metadata.kill[i] else "-"
            g = metadata.gen[i]
            defs_txt.append("k%d: gen {%d}, kill {%s}" % (i, g, kills))
        header = (
            "The control-flow graph has nodes 0..%d (node 0 is the entry, node %d the exit).\n"
            "Edges (from -> to):\n%s"
        ) % (len(metadata.gen) - 1, len(metadata.gen) - 1, ", ".join(lines))
        block = (
            "Each node k%d creates one definition with a unique id and may kill a set of older "
            "definition ids (those no longer valid past it). Node k%d:\n%s\n\n"
            "Compute the set of definition ids that reach the entry of node %d.\n"
            "A definition reaches a point if some execution path from its creating node to that "
            "point never passes through a node that kills it. At the entry of node 0 this set is empty."
            % (0, 0, "\n".join(defs_txt), metadata.query))
        prompt = header + "\n\n" + block
        prompt += ("\n\nGive the answer as the definition ids in the reaching set, space-separated in "
                   "increasing order. If the set is empty, answer \"-\".")
        return prompt

    def score_answer(self, answer, entry):
        gold = entry.answer
        ans = answer.strip()
        if ans == gold:
            return 1.0
        if ans == "":
            return 0.0
        if ans == "-":
            return 1.0 if gold == "-" else 0.0
        try:
            a = sorted(int(x) for x in ans.split())
        except ValueError:
            return 0.0
        try:
            g = sorted(int(x) for x in gold.split())
        except ValueError:
            g = []
        return 1.0 if a == g else 0.0
