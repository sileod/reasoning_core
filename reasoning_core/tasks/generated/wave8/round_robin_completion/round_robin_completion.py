import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'round_robin_completion (draw 1 of 2)',
 'hypothesis': 'W1-050',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/round_robin_completion',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 617708580,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def compute_completion_order(arrivals, bursts, quantum):
    n = len(arrivals)
    remaining = list(bursts)
    time = 0
    order = []
    while True:
        progressed = False
        for i in range(n):
            if remaining[i] <= 0:
                continue
            if time < arrivals[i]:
                time = arrivals[i]
            progressed = True
            if remaining[i] <= quantum:
                time += remaining[i]
                remaining[i] = 0
                order.append(i + 1)
            else:
                time += quantum
                remaining[i] -= quantum
        if not progressed:
            break
    return order


@dataclass
class RoundRobinCompletionConfig(Config):
    n_proc: int = 4
    max_burst: int = 6
    max_arrival: int = 4
    quantum: int = 2

    def apply_difficulty(self, level):
        self.n_proc = sround(self.n_proc + level)
        self.max_burst = sround(self.max_burst + level)
        self.max_arrival = sround(self.max_arrival + level)
        self.quantum = sround(self.quantum)


class RoundRobinCompletion(Task):
    summary = ("Given arrivals, burst lengths, and quantum, output process "
               "completion order under Round Robin, with per-process bursts and "
               "staggered arrivals over varied process counts and quantum sizes.")
    config_cls = RoundRobinCompletionConfig

    def generate_entry(self):
        cfg = self.config
        n = cfg.n_proc
        quantum = cfg.quantum
        arrivals = [random.randint(0, cfg.max_arrival) for _ in range(n)]
        bursts = [random.randint(1, cfg.max_burst) for _ in range(n)]
        order = compute_completion_order(arrivals, bursts, quantum)
        metadata = edict({
            "arrivals": arrivals,
            "bursts": bursts,
            "quantum": quantum,
            "completion_order": order,
        })
        metadata.payload = {
            "arrivals": arrivals,
            "bursts": bursts,
            "quantum": quantum,
        }
        answer = ",".join(str(x) for x in order)
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        proc_list = ", ".join(
            "P%d: arrive t=%d, burst=%d" % (i + 1, a, b)
            for i, (a, b) in enumerate(zip(metadata.arrivals, metadata.bursts))
        )
        return (
            f"Scheduling {len(metadata.arrivals)} processes with a Round Robin "
            f"CPU scheduler with time quantum {metadata.quantum}.\n"
            f"{proc_list}.\n"
            "Processes are scheduled in increasing index order at each round. A process "
            "runs for up to one quantum when it arrives (or is ready), then yields; if it "
            "finishes within its quantum, it completes at that moment and is output. "
            "Idle time passes when no process is ready.\n"
            "Output the process completion order as process IDs separated by commas "
            "(e.g. '2,1,3')."
        )

    def score_answer(self, answer, entry):
        truth = entry.metadata.completion_order
        try:
            parts = [p.strip() for p in answer.split(",") if p.strip()]
            parsed = [int(p) for p in parts]
        except (ValueError, TypeError):
            return 0.0
        if parsed == truth:
            return 1.0
        return 0.0
