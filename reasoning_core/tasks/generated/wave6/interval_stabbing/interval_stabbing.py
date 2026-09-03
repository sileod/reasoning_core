import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

SCRIPTS = {
    "deliveries": [
        "package handoff", "courier pickup", "sorting-bay slot", "driver swap",
        "loading-dock slot", "return pickup", "depot refuel stop", "parcel drop",
    ],
    "rooms": [
        "interview slot", "meeting slot", "review slot", "video-call slot",
        "workshop slot", "demo slot", "feedback slot", "onboarding slot",
    ],
    "guard": [
        "patrol leg", "checkpoint round", "inspection sweep", "lux-density sweep",
        "gate review", "perimeter pass", "observation run", "shift review",
    ],
}


def _greedy_stab(intervals):
    s = sorted(intervals, key=lambda iv: iv[1])
    points = []
    for a, b in s:
        if not points or points[-1] < a:
            points.append(b)
    return points


def _brute_min(intervals):
    ends = {b for _, b in intervals}
    best = None
    for m in range(0, len(ends) + 1):
        import itertools
        for combo in itertools.combinations(sorted(ends), m):
            if all(any(a <= p <= b for p in combo) for a, b in intervals):
                return list(combo)
    return None


@dataclass
class IntervalStabbingConfig(Config):
    n: int = 5
    overlap: float = 0.35
    range_: int = 14

    def apply_difficulty(self, level):
        self.n = 4 + sround(level * 1.5)
        self.overlap = float(min(0.9, 0.15 + level * 0.13))
        self.range_ = 12 + sround(level * 4)


class IntervalStabbing(Task):
    config_cls = IntervalStabbingConfig

    def generate_entry(self):
        cfg = self.config
        n = int(cfg.n)
        overlap = float(cfg.overlap)
        r = int(cfg.range_)

        scripts = list(SCRIPTS)
        random.shuffle(scripts)
        chosen = scripts[0]

        for _ in range(200):
            intervals = []
            for _i in range(n):
                lo = random.randint(0, r)
                hi = lo + random.randint(1, max(3, r // 3))
                intervals.append((int(lo), int(hi)))
            greedy = _greedy_stab(intervals)
            brute = _brute_min(intervals)
            assert brute is not None
            if len(greedy) == len(brute):
                break
        else:
            raise RuntimeError("could not construct instance")

        assert len(greedy) == len(brute), "greedy not minimal"
        for a, b in intervals:
            assert any(a <= p <= b for p in greedy), "interval unstabbed"

        names = random.choices(SCRIPTS[chosen], k=n)
        clock = random.randint(5, 10)
        abs_intervals = [(a + clock, b + clock) for a, b in intervals]
        abs_greedy = [p + clock for p in greedy]
        lines = [f"There are {n} intervals on a single number line running from {clock}:00 to {clock + r + 2}:00."]
        for i in range(n):
            a, b = abs_intervals[i]
            lines.append(f"{i+1}. {names[i]} from {a}:00 to {b}:00.")
        lines.append(
            "Pick the smallest set of integer times (in hours) so that every interval "
            "contains at least one chosen time. Ties break toward the smallest point. "
            "Give the chosen times as a comma-separated increasing list, e.g. 9,12."
        )

        answer = ",".join(str(p) for p in abs_greedy)
        payload = {"text": "\n".join(lines)}
        metadata = edict({
            "intervals": abs_intervals,
            "greedy": abs_greedy,
            "brute": brute,
            "token": chosen,
        })
        metadata.payload = payload
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return render_payload(metadata.payload)

    def score_answer(self, answer, entry):
        try:
            cleaned = str(answer).strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                cleaned = cleaned[1:-1]
            parts = [p.strip() for p in cleaned.split(",") if p.strip() != ""]
            got = [int(p) for p in parts]
        except Exception:
            return 0.0
        gold = [int(p) for p in str(entry.answer).split(",")]
        if len(got) != len(gold):
            return 0.0
        intervals = entry.metadata.intervals
        if sorted(got) != got:
            return 0.0
        if not all(any(a <= p <= b for p in got) for a, b in intervals):
            return 0.0
        return 1.0 if got == gold else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'Minimum set of points stabbing every interval.',
 'hypothesis': 'S59',
 'changes': 'New task; the answer is a list whose length varies with the '
            'instance.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 348479496,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
