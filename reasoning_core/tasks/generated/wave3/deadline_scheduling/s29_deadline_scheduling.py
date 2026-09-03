import random
from dataclasses import dataclass

from reasoning_core.template import Entry, Config, Task, edict


@dataclass
class DeadlineSchedulingConfig(Config):
    n_jobs: float = 11
    max_time: float = 4
    max_deadline: float = 60

    def apply_difficulty(self, level):
        self.n_jobs = self.n_jobs + 1.2 * level
        self.max_time = self.max_time + 0.2 * level
        self.max_deadline = self.max_deadline + 4 * level


def _moore_hodgson(jobs):
    jobs = sorted(jobs, key=lambda j: j[1])
    selected = []
    total = 0
    for time, deadline in jobs:
        selected.append((time, deadline))
        total += time
        if total > deadline:
            drop = max(selected, key=lambda j: j[0])
            selected.remove(drop)
            total -= drop[0]
    return len(selected)


class DeadlineScheduling(Task):
    summary = "Count the most jobs that can each finish by their deadline on one non-preemptive machine."
    config_cls = DeadlineSchedulingConfig

    def generate_entry(self):
        cfg = self.config
        n = int(cfg.n_jobs)
        upper = n
        lower = 0
        for _ in range(400):
            times = [random.randint(1, max(2, int(cfg.max_time))) for _ in range(n)]
            total = sum(times)
            relax = random.uniform(0.15, 1.0)
            horizon = max(1, int(total * relax))
            deadlines = [random.randint(t, max(t, horizon)) for t in times]
            jobs = sorted((t, d) for t, d in zip(times, deadlines))
            best = _moore_hodgson(jobs)
            if lower < best < upper:
                return Entry(
                    metadata=edict(times=times, deadlines=deadlines, n_jobs=n),
                    answer=str(best),
                )
        raise RuntimeError("Could not find non-saturated deadline instance")

    def render_prompt(self, m):
        lines = "\n".join(
            f"  Job {i+1}: processing time {t}, deadline {d}"
            for i, (t, d) in enumerate(zip(m.times, m.deadlines))
        )
        return (
            "On a single machine with no preemption, jobs are scheduled one at a time; "
            "a job finishes on time if it completes no later than its deadline. Jobs:\n"
            f"{lines}\n"
            "The answer is the largest number of jobs that can all finish on time."
        )

    def score_answer(self, answer, entry):
        return float(str(answer).strip() == str(entry.answer).strip())


TASK_META = {'parent_source_id': None,
 'idea': 'Add deadline scheduling on a single machine.',
 'hypothesis': 'S29',
 'changes': 'Ask how many jobs can meet their deadlines, or the minimum total '
            'lateness.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3986056564,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
