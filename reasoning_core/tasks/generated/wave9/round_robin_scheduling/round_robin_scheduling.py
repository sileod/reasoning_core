from dataclasses import dataclass, field
import random

from reasoning_core.template import Task, Entry, Config, edict


TASK_META = {'parent_source_id': None,
 'idea': 'round_robin_scheduling (draw 1 of 1)',
 'hypothesis': 'HV-044',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/round_robin_scheduling',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2693117635,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _simulate(n, quantum, arrivals, bursts):
    remaining = list(bursts)
    running = [0] * n
    completion = [None] * n
    ready = []
    arrived_upto = 0
    t = 0
    done = 0
    while done < n:
        while arrived_upto < n and arrivals[arrived_upto] <= t:
            ready.append(arrived_upto)
            arrived_upto += 1
        if not ready:
            t = min(arrivals[i] for i in range(n) if completion[i] is None)
            continue
        p = ready.pop(0)
        qt = quantum if remaining[p] > quantum else remaining[p]
        remaining[p] -= qt
        t += qt
        running[p] += 1
        while arrived_upto < n and arrivals[arrived_upto] <= t:
            ready.append(arrived_upto)
            arrived_upto += 1
        if remaining[p] == 0:
            completion[p] = t
            done += 1
        else:
            ready.append(p)
    waiting = [completion[i] - arrivals[i] - bursts[i] for i in range(n)]
    return running, waiting, completion


@dataclass
class RoundRobinSchedulingConfig(Config):
    n_procs: int = 4
    max_burst: int = 14
    max_arrival: int = 8
    shift: int = 0
    task_version: int = 2

    def apply_difficulty(self, level):
        self.n_procs = int(round(3 + 0.8 * level))
        self.max_burst = int(round(8 + 1.6 * level))
        self.max_arrival = int(round(4 + 1.0 * level))
        self.shift = int(round(0.7 * level))

    def _quantum(self):
        return max(2, int(round(3 + 0.3 * self.max_burst)))


class RoundRobinScheduling(Task):
    summary = ("Simulate process arrivals and CPU bursts under fixed-quantum round-robin "
               "scheduling, returning a queried execution, waiting, or completion result.")
    config_cls = RoundRobinSchedulingConfig

    def generate_entry(self):
        c = self.config
        n = c.n_procs
        quantum = c._quantum()

        chosen = random.randint(0, 2)
        label = ("executions", "waiting", "completions")[chosen]

        arrivals = [0] * n
        for i in range(1, n):
            arrivals[i] = random.randint(0, c.max_arrival)
        arrivals = [min(max(a, 0), c.max_arrival) for a in arrivals]
        arrivals.sort()
        arrivals[0] = 0

        bursts = [random.randint(1, c.max_burst) for _ in range(n)]

        running, waiting, completion = _simulate(n, quantum, arrivals, bursts)

        if chosen == 0:
            target = random.randint(0, n - 1)
            answer = running[target]
            query = f"how many times process P{target} is assigned to the CPU"
        elif chosen == 1:
            target = random.randint(0, n - 1)
            answer = waiting[target]
            query = f"the total waiting time (in time units) of process P{target}"
        else:
            target = random.randint(0, n - 1)
            answer = completion[target]
            query = f"the completion time (in time units, when it first finishes) of process P{target}"

        metadata = edict({
            "n": n,
            "quantum": quantum,
            "arrivals": arrivals,
            "bursts": bursts,
            "chosen": label,
            "target": target,
            "answer": int(answer),
        })
        metadata.payload = {
            "n": n,
            "quantum": quantum,
            "arrivals": [int(a) for a in arrivals],
            "bursts": [int(b) for b in bursts],
        }

        return Entry(metadata=metadata, answer=str(int(answer)))

    def render_prompt(self, metadata):
        p = metadata.payload
        lines = []
        lines.append(
            f"We run fixed-quantum round-robin scheduling with quantum {p['quantum']} time units "
            f"on {p['n']} single-CPU processes."
        )
        lines.append("Arrival time, burst time:")
        procs = ", ".join(
            f"(P{i}: arrives at {p['arrivals'][i]}, burst {p['bursts'][i]})"
            for i in range(len(p['arrivals']))
        )
        lines.append(procs)
        lines.append(
            "The CPU serves the ready queue in FIFO order, each ready process getting at most "
            "one quantum of CPU before returning to the tail of the queue; a newly arrived "
            "process joins at the tail. A process that finishes its remaining burst before "
            "exhausting a quantum leaves the schedule."
        )
        lines.append(f"Report {metadata.chosen}: {metadata.chosen} for process P{metadata.target}.")
        return "\n".join(lines) + "\n\nThe answer is a non-negative integer."

    def score_answer(self, answer, entry):
        try:
            val = int(str(answer).strip())
        except Exception:
            return 0.0
        return 1.0 if val == int(entry.answer) else 0.0
