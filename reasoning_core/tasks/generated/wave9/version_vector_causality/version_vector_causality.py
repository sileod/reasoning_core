import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'version_vector_causality (draw 1 of 1)',
 'hypothesis': 'HV-048',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/version_vector_causality',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2951849487,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class VectorCausalityConfig(Config):
    n_procs: int = 4
    n_events: int = 8
    def apply_difficulty(self, level):
        self.n_procs = sround(self.n_procs + level)
        self.n_events = sround(self.n_events + 2 * level)


class VersionVectorCausality(Task):
    summary = "Execute local, send, and receive events using vector clocks, then return the final vector-clock value at a queried event as a comma-separated integer list."
    config_cls = VectorCausalityConfig
    task_version = 2

    def generate_entry(self):
        cfg = self.config
        n = int(cfg.n_procs)
        n_ev = int(cfg.n_events)
        for _ in range(400):
            procs = list(range(n))
            per_proc_idx = {p: 0 for p in procs}
            ev_list = []
            for _ in range(n_ev):
                pid = random.choice(procs)
                idx = per_proc_idx[pid]
                per_proc_idx[pid] += 1
                ev_list.append((pid, idx))
            ev_proc = {eid: pid for eid, (pid, _) in enumerate(ev_list)}
            events_types = ['local'] * n_ev
            mapping = {}
            n_send = random.randint(1, max(1, n_ev // 3))
            n_send = min(n_send, n_ev)
            send_ids = set(random.sample(range(n_ev), n_send))
            used_recv = set()
            for sid in sorted(send_ids):
                candidates = [eid for eid in range(n_ev)
                              if eid > sid and ev_proc[eid] != ev_proc[sid]
                              and eid not in used_recv]
                if not candidates:
                    continue
                rid = random.choice(candidates)
                used_recv.add(rid)
                mapping[sid] = rid
                events_types[sid] = 'send'
                events_types[rid] = 'receive'
            q1 = random.choice(range(n_ev))
            vc_snapshot = [[0] * n for _ in range(n)]
            vc = []
            for eid in range(n_ev):
                pid = ev_proc[eid]
                t = events_types[eid]
                if t == 'local':
                    v = list(vc_snapshot[pid])
                    v[pid] += 1
                    vc_snapshot[pid] = list(v)
                    vc.append(list(v))
                elif t == 'send':
                    v = list(vc_snapshot[pid])
                    v[pid] += 1
                    vc_snapshot[pid] = list(v)
                    vc.append(list(v))
                else:
                    sid = [k for k, v in mapping.items() if v == eid][0]
                    merged = [max(a, b) for a, b in zip(vc_snapshot[pid], vc[sid])]
                    merged[pid] = vc_snapshot[pid][pid] + 1
                    vc_snapshot[pid] = list(merged)
                    vc.append(list(merged))
            cq = vc[q1]
            lines = []
            for eid in range(n_ev):
                pid = ev_proc[eid]
                t = events_types[eid]
                if t == 'local':
                    lines.append(f"P{pid} executes a local operation (event E{eid}).")
                elif t == 'send':
                    lines.append(f"P{pid} sends a message (event E{eid}).")
                else:
                    lines.append(f"P{pid} receives a message (event E{eid}).")
            answer = ",".join(str(int(x)) for x in cq)
            metadata = edict({
                "n_procs": n,
                "events": lines,
                "q1": q1,
                "vcq": [int(x) for x in cq],
            })
            metadata.payload = {
                "n_procs": n,
                "events": lines,
                "q1": q1,
            }
            return Entry(metadata=metadata, answer=answer)
        raise RuntimeError("Could not build valid instance")

    def render_prompt(self, metadata):
        payload = metadata.payload
        ev = "\n".join(payload["events"])
        return (
            f"There are {metadata.n_procs} processes P0..P{metadata.n_procs-1}, "
            f"each tracking a vector clock of size {metadata.n_procs}.\n\n"
            f"{ev}\n"
            f"\nUsing Lamport-style vector clocks, state the vector clock of event "
            f"E{payload['q1']} at the moment it executes. "
            f"Give the value as a comma-separated list of integers, one per process, "
            f"in order P0, P1, ... .\n\n"
            f"The answer is exactly that comma-separated list."
        )

    def score_answer(self, answer, entry):
        gt = entry.answer
        if not isinstance(answer, str):
            return 0.0
        a = answer.strip().replace(" ", "")
        if a == gt:
            return 1.0
        if a.startswith("[") and a.endswith("]"):
            a = a[1:-1]
        if a == gt:
            return 1.0
        return 0.0
