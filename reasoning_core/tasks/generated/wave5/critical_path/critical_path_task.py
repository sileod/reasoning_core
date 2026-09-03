import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload


TASK_META = {'parent_source_id': None,
 'idea': 'Add critical-path analysis over a small project plan.',
 'hypothesis': 'S56',
 'changes': 'Ask for the project duration and the activities that cannot slip.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2778460392,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


class CriticalPathConfig(Config):
    n_activities: int = 6
    max_wait: int = 3

    def apply_difficulty(self, level):
        self.n_activities = 6 + 2 * level
        self.max_wait = 2 + level


def analyze(n, deps, durations):
    # deps only reference lower indices, so order is 0..n-1.
    earliest = [0] * n
    for v in range(n):
        if deps[v]:
            earliest[v] = max(earliest[d] + durations[d] for d in deps[v])
    project = max(earliest[v] + durations[v] for v in range(n))
    latest = [0] * n
    for v in range(n - 1, -1, -1):
        succ = [u for u in range(v + 1, n) if v in deps[u]]
        if succ:
            latest[v] = min(latest[u] for u in succ) - durations[v]
        else:
            latest[v] = project - durations[v]
    critical = [v for v in range(n) if earliest[v] == latest[v]]
    return project, critical


def _name(i):
    return chr(65 + i)


class CriticalPath(Task):
    config_cls = CriticalPathConfig

    def generate_entry(self):
        n = self.config.n_activities
        max_wait = self.config.max_wait
        for _ in range(1000):
            durations = [100 + random.randint(0, 700) for _ in range(n)]
            deps = [[] for _ in range(n)]
            for u in range(n):
                ndep = random.randint(0, min(max_wait, u))
                pool = list(range(u))
                random.shuffle(pool)
                deps[u] = sorted(pool[:ndep])
            project, critical = analyze(n, deps, durations)
            if len(critical) >= 2:
                break
        else:
            raise RuntimeError("could not build instance with >=2 critical activities")

        question = random.choice(['duration', 'critical'])
        metadata = edict({
            'durations': [int(d) for d in durations],
            'deps': [[int(x) for x in d] for d in deps],
            'critical': [int(c) for c in critical],
            'project': float(project),
            'question': question,
        })
        metadata.payload = {
            'activities': [{'name': _name(i),
                            'duration': int(durations[i]),
                            'waits_for': [_name(d) for d in deps[i]]}
                           for i in range(n)],
        }
        if question == 'duration':
            answer = str(project)
        else:
            answer = ','.join(_name(c) for c in sorted(critical))
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        q = metadata.question
        if q == 'duration':
            ask = ("What is the earliest finish time of the whole project "
                   "(the time at which all activities are complete)?")
        else:
            ask = ("Which activities are critical, that is, have zero slack "
                   "(a delay to any of them delays the whole project)? "
                   "Answer as a comma-separated list in alphabetical order.")
        head = render_payload(metadata.payload)
        return f"{head}\n\n{ask}\n\nReturn only the answer."

    def score_answer(self, answer, entry):
        gold = entry.answer
        q = entry.metadata.question
        if answer is None:
            return 0.0
        a = str(answer).strip()
        if q == 'duration':
            try:
                return 1.0 if float(a) == float(gold) else 0.0
            except ValueError:
                return 0.0
        else:
            parts = sorted(x.strip() for x in a.split(',') if x.strip())
            goldparts = sorted(gold.split(','))
            return 1.0 if parts == goldparts else 0.0
