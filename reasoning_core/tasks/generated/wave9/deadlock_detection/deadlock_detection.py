from dataclasses import dataclass
import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


def _greedy(n, holder, requests):
    """Simulate the canonical tiny-label completion order.

    A process completes when every resource it requests is free, i.e. when no unfinished
    process holds a requested resource.  On completion it releases all resources it holds.
    Returns (completion_sequence, deadlocked_set) where deadlocked_set is the set of
    processes that can never complete.
    """
    held_by = {}
    for res, holder_proc in enumerate(holder):
        held_by.setdefault(holder_proc, set()).add(res)
    requests = [set(r) for r in requests]
    unfin = set(range(n))
    avail = set()
    ordered = []
    while unfin:
        cand = None
        for p in sorted(unfin):
            if requests[p] <= avail:
                cand = p
                break
        if cand is None:
            break
        unfin.discard(cand)
        ordered.append(cand)
        for res in held_by.get(cand, ()):
            avail.add(res)
    return ordered, sorted(unfin)


def _can_complete(n, holder, requests):
    """Independent structural check via SCCs of the wait-for graph.

    p -> j whenever p requests a resource held by j.  A process can complete (under any
    order) iff it cannot reach a non-trivial strongly connected component along wait-for
    edges.  Returns the set of completable processes for cross-checking the greedy.
    """
    adj = [[] for _ in range(n)]
    for res, hp in enumerate(holder):
        for p in range(n):
            if res in requests[p] and p != hp:
                adj[p].append(hp)
    index_counter = [0]
    stack = []
    lowlink = [None] * n
    index = [None] * n
    onstack = [False] * n
    scc_cyclic = set()
    comp_id = [None] * n
    comp_cyclic = {}

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        onstack[v] = True
        for w in adj[v]:
            if index[w] is None:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif onstack[w]:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack[w] = False
                comp.append(w)
                if w == v:
                    break
            cid = len(comp_cyclic)
            cyclic = len(comp) > 1
            for w in comp:
                comp_id[w] = cid
            comp_cyclic[cid] = cyclic

    for v in range(n):
        if index[v] is None:
            strongconnect(v)

    from_to_cyclic = [False] * n
    for p in range(n):
        seen = {p}
        fstack = [p]
        found = False
        while fstack:
            cur = fstack.pop()
            if comp_cyclic[comp_id[cur]]:
                found = True
                break
            for w in adj[cur]:
                if w not in seen:
                    seen.add(w)
                    fstack.append(w)
        from_to_cyclic[p] = found

    completable = [p for p in range(n) if not from_to_cyclic[p]]
    return completable


def _canonical_answer(n, holder, requests):
    ordered, deadlocked = _greedy(n, holder, requests)
    if not deadlocked:
        return ("safe", ordered)
    return ("deadlock", deadlocked)


@dataclass
class DeadlockDetectionConfig(Config):
    n_processes: int = 4
    n_resources: int = 6
    requests_per: int = 2
    cycle_size: int = 3

    def apply_difficulty(self, level):
        self.n_processes = sround(self.n_processes + 2 * level)
        self.n_resources = sround(self.n_resources + 3 * level)
        self.requests_per = sround(self.requests_per + level)
        self.cycle_size = sround(self.cycle_size + level // 2)


class DeadlockDetection(Task):
    summary = "Track resource ownership and waits to construct a wait-for relation, returning the deadlocked process set or a canonical safe sequence."

    config_cls = DeadlockDetectionConfig

    def generate_entry(self):
        cfg = self.config
        n = max(2, cfg.n_processes)
        m = max(1, cfg.n_resources)
        rp = max(1, cfg.requests_per)

        for _ in range(300):
            requests = [[] for _ in range(n)]
            holder = [None] * m
            for r in range(m):
                holder[r] = random.randrange(n)

            flavor = random.random() < 0.5
            if flavor:
                perm = list(range(n))
                random.shuffle(perm)
                pos = [0] * n
                for idx, p in enumerate(perm):
                    pos[p] = idx
                for p in range(n):
                    candidates = [r for r in range(m) if pos[holder[r]] < pos[p]]
                    k = random.randint(0, min(rp, len(candidates)))
                    requests[p] = random.sample(candidates, k)
                kind, answer_src = _canonical_answer(n, holder, requests)
                if kind != "safe":
                    continue
                gold = answer_src
            else:
                sub = random.sample(range(n), max(2, min(n, cfg.cycle_size)))
                if m < len(sub):
                    continue
                for i in range(len(sub)):
                    holder[i] = sub[(i + 1) % len(sub)]
                    requests[sub[i]].append(i)
                base_res = list(range(len(sub), m))
                if base_res:
                    for p in range(n):
                        cands = [r for r in base_res if holder[r] != p]
                        if not cands:
                            continue
                        k = random.randint(0, min(rp, len(cands)))
                        requests[p].extend(random.sample(cands, k))
                kind, answer_src = _canonical_answer(n, holder, requests)
                if kind != "deadlock":
                    continue
                gold = answer_src

            ordered, deadlocked = _greedy(n, holder, requests)
            completable = _can_complete(n, holder, requests)
            if kind == "safe":
                if not (len(gold) == n and not deadlocked and
                        set(completable) == set(range(n))):
                    continue
            else:
                if not (deadlocked and set(deadlocked) ==
                        set(p for p in range(n) if p not in completable)):
                    continue

            labels = sorted(random.sample(range(0, 999), n))
            lab = {p: labels[p] for p in range(n)}
            if kind == "safe":
                answer = ",".join(str(lab[p]) for p in gold)
            else:
                answer = ",".join(str(lab[p]) for p in sorted(gold))
            if len(set(answer.split(","))) < max(2, n // 2):
                continue

            metadata = edict({
                "n": n,
                "m": m,
                "labels": [lab[p] for p in range(n)],
                "holder": list(holder),
                "requests": [list(r) for r in requests],
                "flavor": kind,
            })
            metadata.payload = {
                "query": (
                    f"There are {n} processes, each running on a single-instance resource "
                    "system with a fixed set of resources. Every resource can be held by at "
                    "most one process at a time. Each process currently holds some resources "
                    "and is simultaneously requesting one or more additional resources. A "
                    "process blocks until it obtains every resource it requests at the same "
                    "time; once it does, it finishes and releases all the resources it held. "
                    "A process that requests a resource already held by another process waits "
                    "for that other process to finish and release it."
                ),
                "holders": (
                    "Resource ownership (each row is 'resource: current holder process'):"
                    "\n"
                    + "\n".join(f"R{r}: P{lab[holder[r]]}" for r in range(m))
                ),
                "requests": (
                    "Outstanding requests (each row is 'process: resources it is requesting'):"
                    "\n"
                    + "\n".join(
                        f"P{lab[p]}: {[lab[holder[r]] for r in sorted(requests[p])] or 'none'}"
                        for p in range(n)
                    )
                ),
            }
            return Entry(metadata=metadata, answer=answer)

        raise RuntimeError("could not generate a valid deadlock instance")

    def render_prompt(self, metadata):
        guidance = (
            "Build the wait-for relation from who holds each requested resource: process A "
            "waits on process B when A requests a resource B currently holds. A deadlock "
            "exists when some processes can never obtain their requested resources because "
            "they are waiting (directly or through a chain) on resources held forever by "
            "processes waiting on each other; those processes can never complete.\n\n"
            "If a deadlock exists, the answer is the deadlocked set: every process that can "
            "never complete, written as their labels in ascending order, separated by commas "
            "(for example \"1,3,5\").\n\n"
            "If there is no deadlock, all processes can complete in some order. Produce the "
            "canonical safe sequence by this rule: repeatedly, among processes that have not "
            "yet completed and whose every requested resource is currently free (not held by "
            "any unfinished process), complete the one with the smallest label and release "
            "the resources it held. The answer is that completion order, labels separated by "
            "commas (for example \"0,2,1\").\n\n"
            "The answer is a single comma-separated list of process labels: the ascending "
            "deadlocked set, or the safe completion order."
        )
        return f"{render_payload(metadata.payload)}\n\n{guidance}"

    def score_answer(self, answer, entry):
        return 1.0 if _normalize(answer) == _normalize(entry.answer) else 0.0


def _normalize(text):
    return " ".join(str(text).split()).strip()


TASK_META = {'parent_source_id': None,
 'idea': 'deadlock_detection (draw 1 of 1)',
 'hypothesis': 'HV-050',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/deadlock_detection',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 4249108675,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
