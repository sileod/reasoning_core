import random
from dataclasses import dataclass
from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'event_queue_simulation (draw 1 of 1)',
 'hypothesis': 'HV-043',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/event_queue_simulation',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 4052223613,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class EventQueueSimConfig(Config):
    n_procs: int = 3
    n_events: int = 6
    horizon: int = 30

    def apply_difficulty(self, level):
        self.n_procs = sround(self.n_procs + level)
        self.n_events = sround(self.n_events + 2 * level)
        self.horizon = sround(self.horizon + 8 * level)


class EventQueueSimulation(Task):
    summary = "Execute timestamped discrete events whose handlers schedule later events under stated priority rules, returning a queried terminal system state."

    config_cls = EventQueueSimConfig

    def generate_entry(self):
        cfg = self.config
        n_procs = cfg.n_procs
        n_events = cfg.n_events
        horizon = cfg.horizon

        arrivals = []
        for _ in range(n_events):
            arr_time = random.randint(0, horizon)
            proc = random.randint(0, n_procs - 1)
            work = random.randint(1, 6)
            priority = random.randint(1, 5)
            arrivals.append([arr_time, proc, work, priority])

        # Index each arrival for tie-breaking.
        idxed = [(t, p, w, pr, i) for i, (t, p, w, pr) in enumerate(arrivals)]

        # Simulation: per-processor work queue.
        # Event timeline: we process arrivals in global time order. When a job
        # arrives at time t for proc p with work w and priority pr, it enters the
        # processor's pending queue. A processor is free at time F_p. When free,
        # it starts the highest-priority (lowest number, then earliest arrival index)
        # pending job. Start time = max(arrival, F_p); completion = start + work.
        # We run a discrete-event simulation to compute, for each processor, when
        # all its jobs finish, then take the max as the total completion time.

        # Build per-proc job list with arrival times.
        jobs = {}
        for t, p, w, pr, i in idxed:
            jobs.setdefault(p, []).append((t, w, pr, i))

        total_completion = 0
        for p, plist in jobs.items():
            # Sort by (priority, arrival_index) = processing order.
            order = sorted(plist, key=lambda j: (j[2], j[3]))
            free = 0
            for (t, w, pr, i) in order:
                start = max(free, t)
                free = start + w
            total_completion = max(total_completion, free)

        # Domain check: non-negative integer.
        assert isinstance(total_completion, int) and total_completion >= 0

        answer = str(total_completion)

        metadata = edict({
            "n_procs": n_procs,
            "arrivals": [list(a[:4]) for a in idxed],
        })
        metadata.payload = {
            "n_procs": n_procs,
            "arrivals": [list(a[:4]) for a in idxed],
        }

        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = []
        lines.append(f"There are {metadata.n_procs} numbered processors. Jobs arrive at an event queue over time; each arrival states a time, a target processor, a work amount, and a priority where a lower number means higher priority. Ties in priority are broken by order of arrival into the queue.")
        for (t, p, w, pr) in metadata.arrivals:
            lines.append(f"At time {t}, a job for processor {p} arrives needing {w} units of work with priority {pr}.")
        lines.append("Each processor processes one job at a time. When it becomes free, it starts the pending job of highest priority (ties by earlier arrival). A job starts no earlier than its arrival time. When there are no jobs left for a processor, that processor is done.")
        lines.append("What is the total completion time, defined as the maximum over all processors of the time when a processor finishes all of its jobs?")
        lines.append("The answer is a non-negative integer.")
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        try:
            val = int(str(answer).strip())
        except Exception:
            return 0.0
        gold = int(entry.answer)
        return 1.0 if val == gold else 0.0
