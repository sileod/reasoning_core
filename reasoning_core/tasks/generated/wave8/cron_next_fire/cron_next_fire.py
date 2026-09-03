import calendar
import random
from dataclasses import dataclass
from datetime import datetime

from reasoning_core.template import Config, Entry, Task, edict, render_payload

TASK_META = {'parent_source_id': None,
 'idea': 'cron_next_fire (draw 1 of 2)',
 'hypothesis': 'W1-076',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/cron_next_fire',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 901858890,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _render(fields):
    labels = ["minute", "hour", "day-of-month", "month"]
    parts = []
    for name, val in zip(labels, fields):
        parts.append(f"{name}={val if val is not None else '*'}")
    return ", ".join(parts)


def _next_fire(fields, ref_dt):
    target_min, target_hour, target_dom, target_month = fields
    months = [target_month] if target_month is not None else list(range(1, 13))
    hours = [target_hour] if target_hour is not None else list(range(24))
    minutes = [target_min] if target_min is not None else list(range(60))
    best = None
    ref_date = ref_dt.date()
    for y in range(ref_dt.year, ref_dt.year + 2):
        for mo in months:
            dim = calendar.monthrange(y, mo)[1]
            doms = [target_dom] if target_dom is not None else list(range(1, dim + 1))
            for dom in doms:
                if dom > dim:
                    continue
                d = datetime(y, mo, dom)
                if d.date() < ref_date:
                    continue
                if d.date() == ref_date:
                    for h in hours:
                        for m in minutes:
                            cand = d.replace(hour=h, minute=m)
                            if cand <= ref_dt:
                                continue
                            if best is None or cand < best:
                                best = cand
                else:
                    cand = d.replace(hour=min(hours), minute=min(minutes))
                    if best is None or cand < best:
                        best = cand
    return best


@dataclass
class CronNextFireConfig(Config):
    fixed_count: int = 2

    def apply_difficulty(self, level):
        self.fixed_count = self.fixed_count + level


class CronNextFire(Task):
    summary = ("Given a restricted numeric cron expression (minute, hour, day-of-month, "
               "month, each a single value or wildcard) and a reference timestamp, output "
               "the next matching timestamp strictly after it; difficulty scales the number "
               "of fixed fields so the gap spans minutes to nearly a year.")
    config_cls = CronNextFireConfig
    task_version = 2

    def generate_entry(self):
        n_fixed = int(self.config.fixed_count)
        n_fixed = max(1, min(4, n_fixed))
        fields = [None, None, None, None]
        fixed_fields = set(random.sample(range(4), n_fixed))
        for i in fixed_fields:
            if i == 0:
                fields[i] = random.randint(0, 59)
            elif i == 1:
                fields[i] = random.randint(0, 23)
            elif i == 3:
                fields[i] = random.randint(1, 12)
            else:
                fields[i] = random.randint(1, 31)
        if fields[3] is not None and fields[2] is not None:
            dim = 28 if fields[3] == 2 else calendar.monthrange(2024, fields[3])[1]
            if fields[2] > dim:
                fields[2] = random.randint(1, dim)

        ref_dt = datetime(
            random.randint(2020, 2035),
            random.randint(1, 12),
            random.randint(1, 28),
            random.randint(0, 23),
            random.randint(0, 59),
        )
        nxt = _next_fire(fields, ref_dt)
        if nxt is None:
            raise RuntimeError("no next fire found")
        assert nxt > ref_dt
        assert _next_fire(fields, nxt) is not None  # domain reachable / consistent

        answer = nxt.strftime("%Y-%m-%d %H:%M")
        cron_str = _render(fields)
        ts_str = ref_dt.strftime("%Y-%m-%d %H:%M")
        metadata = edict({
            "payload": {
                "cron expression": cron_str,
                "reference timestamp": ts_str,
            }
        })
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        body = render_payload(metadata.payload)
        return (
            f"{body}\n\n"
            "A cron-like expression lists a single value (or * for 'any') for each of the "
            "fields minute, hour, day-of-month and month. A timestamp matches when every "
            "numbered field equals its meaning (day-of-month 1-31, month 1-12); * matches "
            "any value.\n\n"
            "Write the earliest timestamp strictly after the reference timestamp that "
            "matches the expression. Give it in the format YYYY-MM-DD HH:MM (a worked "
            "example: 2031-04-07 09:14).\n\n"
            "The answer is the matching timestamp:"
        )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        return 1.0 if answer.strip() == str(entry.answer) else 0.0
