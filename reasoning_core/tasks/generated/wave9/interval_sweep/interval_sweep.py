import random
import ast
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'interval_sweep (draw 1 of 1)',
 'hypothesis': 'HV-020',
 'changes': 'new task in reasoning_core/tasks/generated/wave9/interval_sweep',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2068888396,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class IntervalSweepConfig(Config):
    n_intervals: int = 5
    coord_range: int = 24
    mode: str = "peak"

    def apply_difficulty(self, level):
        self.n_intervals = sround(self.n_intervals + level * 2)
        self.coord_range = sround(self.coord_range + level * 10)
        if level < 2:
            self.mode = "peak"
        elif level < 4:
            self.mode = "merge"
        else:
            self.mode = "active"


def _gen_interval(coord_range, rng=random):
    a = rng.randint(0, coord_range)
    b = rng.randint(0, coord_range)
    if b < a:
        a, b = b, a
    return (a, b)


def _merge(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            if b > merged[-1][1]:
                merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged]


class IntervalSweep(Task):
    summary = "Execute ordered endpoint sweeps over intervals to compute peak overlap, covered regions (merged union length or canonical merged intervals), or active overlapping intervals at a query point."
    config_cls = IntervalSweepConfig
    task_version = 2

    def generate_entry(self):
        cfg = self.config
        n = cfg.n_intervals
        coord_range = cfg.coord_range
        mode = cfg.mode

        while True:
            intervals = [tuple(_gen_interval(coord_range)) for _ in range(n)]
            if mode == "peak":
                events = []
                for a, b in intervals:
                    events.append((a, 1))
                    events.append((b, -1))
                events.sort(key=lambda x: (x[0], x[1]))
                cur = 0
                peak = 0
                for _, d in events:
                    cur += d
                    if cur > peak:
                        peak = cur
                assert 0 <= peak <= n
                if peak <= 1 or peak >= n:
                    continue
                answer = str(peak)
                label = answer
            elif mode == "merge":
                merged = _merge(intervals)
                if len(merged) < 2:
                    continue
                answer = "; ".join(f"{a}-{b}" for a, b in merged)
                label = answer
            else:
                q = random.randint(0, coord_range)
                active = sorted(a for a, b in intervals if a <= q <= b)
                if len(active) < 2:
                    continue
                answer = "; ".join(str(v) for v in active)
                label = answer

            metadata = edict({
                "intervals": intervals,
                "mode": mode,
                "query": q if mode == "active" else None,
                "answer": answer,
            })
            metadata.payload = {
                "intervals": intervals,
                "mode": mode,
                "query": q if mode == "active" else None,
            }
            return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        mode = metadata.mode
        if mode == "peak":
            return (
                f"Consider the following set of intervals on the number line, each given as "
                f"(start, end) inclusive of both endpoints: {render_payload({'intervals': metadata.intervals})}. "
                f"Perform a sweep over the endpoints, counting how many intervals are active "
                f"as you move left to right, and report the maximum number of intervals that "
                f"overlap at any single point. The answer is a single integer (the peak overlap)."
            )
        elif mode == "merge":
            return (
                f"Consider the following set of intervals on the number line, each given as "
                f"(start, end) inclusive of both endpoints: {render_payload({'intervals': metadata.intervals})}. "
                f"Merge all overlapping or touching intervals into disjoint canonical intervals. "
                f"Report the merged intervals, each as start-end, separated by semicolons, "
                f"in increasing order of start. Example of the answer format: 2-5; 8-10."
            )
        else:
            return (
                f"Consider the following set of intervals on the number line, each given as "
                f"(start, end) inclusive of both endpoints: {render_payload({'intervals': metadata.intervals})}. "
                f"The query point is {metadata.query}. Sweep to find every start value of the "
                f"intervals that contain the query point. Report those start values in increasing "
                f"order, separated by semicolons. Example of the answer format: 2; 8; 9"
            )

    def score_answer(self, answer, entry):
        try:
            gold = entry.answer
        except Exception:
            return 0.0
        if not isinstance(answer, str):
            return 0.0
        a = answer.strip()
        if mode_is_peak(entry):
            return 1.0 if a == gold else 0.0
        if mode_is_active(entry):
            if a == gold:
                return 1.0
            try:
                plist = [int(x.strip()) for x in a.split(";")]
            except Exception:
                return 0.0
            try:
                glist = [int(x.strip()) for x in gold.split(";")]
            except Exception:
                return 0.0
            return 1.0 if plist == glist else 0.0
        return 1.0 if a == gold else 0.0


def mode_is_peak(entry):
    try:
        return entry.metadata.mode == "peak"
    except Exception:
        return False


def mode_is_active(entry):
    try:
        return entry.metadata.mode == "active"
    except Exception:
        return False
