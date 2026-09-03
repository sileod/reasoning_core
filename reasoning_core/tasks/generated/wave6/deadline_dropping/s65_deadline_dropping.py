import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'Which jobs finish on time when late ones must be dropped.',
 'hypothesis': 'S65',
 'changes': 'New task; the answer is the on-time set, verified by simulation.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 65042966,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _moore(durations, deadlines, names):
    order = sorted(range(len(names)), key=lambda i: deadlines[i])
    accepted = []
    total = 0
    for i in order:
        accepted.append(i)
        total += durations[i]
        if total > deadlines[i]:
            longest = max(range(len(accepted)), key=lambda k: durations[accepted[k]])
            total -= durations[accepted[longest]]
            accepted.pop(longest)
    return sorted(accepted)


def _largest_set_size(durations, deadlines, names):
    n = len(names)
    best = 0
    for mask in range(1 << n):
        chosen = [i for i in range(n) if mask & (1 << i)]
        if not chosen:
            continue
        chosen_sorted = sorted(chosen, key=lambda i: deadlines[i])
        t = 0
        ok = True
        for i in chosen_sorted:
            t += durations[i]
            if t > deadlines[i]:
                ok = False
                break
        if ok:
            best = max(best, len(chosen))
    return best


@dataclass
class DeadlineDroppingConfig(Config):
    n_jobs: int = 5
    max_deadline: int = 14
    tight: bool = False

    def apply_difficulty(self, level):
        self.n_jobs = sround(self.n_jobs + level)
        self.max_deadline = sround(self.max_deadline + 2 * level)
        self.tight = bool(level >= 3)


class DeadlineDropping(Task):
    config_cls = DeadlineDroppingConfig

    def generate_entry(self):
        cfg = self.config
        while True:
            n = cfg.n_jobs
            names = []
            used = set()
            for _ in range(n):
                while True:
                    nm = random.randint(100, 999)
                    if nm not in used:
                        used.add(nm)
                        names.append(nm)
                        break
            durations = [random.randint(2, 6) for _ in range(n)]
            if cfg.tight:
                deadlines = [random.randint(5, 9) for _ in range(n)]
            else:
                deadlines = [random.randint(8, cfg.max_deadline) for _ in range(n)]
            durations = [min(d, d_max) for d, d_max in zip(durations, deadlines)]
            durations = [d for d in durations]
            if any(d > 0 for d in durations) and min(deadlines) >= 2:
                base = _moore(durations, deadlines, names)
                size = len(base)
                if durations and all(max(durations) < min(deadlines) or True for _ in [0]):
                    pass
                biggest = _largest_set_size(durations, deadlines, names)
                if biggest == size and size >= 2:
                    order = sorted(range(n), key=lambda i: deadlines[i])
                    t = 0
                    ok = True
                    verify = []
                    for i in order:
                        if i in base:
                            t += durations[i]
                            if t > deadlines[i]:
                                ok = False
                                break
                            verify.append(names[i])
                    if ok:
                        total = _largest_set_size(durations, deadlines, names)
                        dedup = self._check_unique(durations, deadlines, names, size)
                        if dedup:
                            self._simulate(durations, deadlines, names, base)
                            break
        names_str = ", ".join(str(x) for x in names)
        ans_order = sorted(verify, key=lambda nm: deadlines[names.index(nm)])
        answer = ", ".join(str(x) for x in ans_order)
        metadata = edict({
            "n_jobs": n,
            "names": names,
            "durations": durations,
            "deadlines": deadlines,
            "answer": answer,
        })
        metadata.payload = {
            "names": names,
            "durations": durations,
            "deadlines": deadlines,
        }
        return Entry(metadata=metadata, answer=answer)

    def _check_unique(self, durations, deadlines, names, size):
        count = 0
        for mask in range(1 << len(names)):
            chosen = [i for i in range(len(names)) if mask & (1 << i)]
            if len(chosen) != size:
                continue
            chosen_sorted = sorted(chosen, key=lambda i: deadlines[i])
            t = 0
            ok = True
            for i in chosen_sorted:
                t += durations[i]
                if t > deadlines[i]:
                    ok = False
                    break
            if ok:
                count += 1
        return count == 1

    def _simulate(self, durations, deadlines, names, base):
        order = sorted(range(len(names)), key=lambda i: deadlines[i])
        t = 0
        for i in order:
            if i in base:
                t += durations[i]
                assert t <= deadlines[i]

    def render_prompt(self, metadata):
        pairs = []
        for nm, d, dl in zip(metadata.names, metadata.durations, metadata.deadlines):
            pairs.append(f"job {nm}: duration {d}, deadline {dl}")
        body = "; ".join(pairs)
        return (f"The following jobs are all available now, and the machine runs one at a time. "
                f"{body}. Find the largest set of jobs that can all finish by their deadlines. "
                f"Exactly one set of that size works. "
                f"The answer is a comma-separated list of job names ordered by deadline.")

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        try:
            try:
                nums = [int(x.strip()) for x in answer.split(",") if x.strip()]
            except ValueError:
                return 0.0
            truth = [int(x.strip()) for x in entry.answer.split(",") if x.strip()]
        except Exception:
            return 0.0
        return 1.0 if nums == truth else 0.0
