import calendar
import random
from dataclasses import dataclass
from datetime import date, timedelta

from reasoning_core.template import Task, Entry, Config, edict

TASK_META = {'parent_source_id': None,
 'idea': 'Add reasoning over calendar recurrence rules.',
 'hypothesis': 'S55',
 'changes': 'Ask for the date of the nth occurrence of a recurring event.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 342837742,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                 "Saturday", "Sunday"]
MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November",
               "December"]


def _ordinal(x):
    if 10 <= x % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(x % 10, "th")
    return "{}{}".format(x, suffix)


def _add_months(y, m, k):
    tot = y * 12 + (m - 1) + k
    return tot // 12, tot % 12 + 1


def _month_date(y, m, spec):
    if spec[0] == "nth":
        _, o, wd = spec
        first = date(y, m, 1)
        delta = (wd - first.weekday()) % 7
        return date(y, m, 1 + delta + (o - 1) * 7)
    if spec[0] == "last":
        _, wd = spec
        lastd = calendar.monthrange(y, m)[1]
        last = date(y, m, lastd)
        return last - timedelta(days=(last.weekday() - wd) % 7)
    return date(y, m, calendar.monthrange(y, m)[1])


def _business(d):
    if d.weekday() >= 5:
        return d + timedelta(days=7 - d.weekday())
    return d


def _fmt_months(k):
    return "every month" if k == 1 else "every {} months".format(k)


@dataclass
class CalendarRecurrenceConfig(Config):
    def apply_difficulty(self, level):
        self.level = level


class CalendarRecurrence(Task):
    config_cls = CalendarRecurrenceConfig

    def generate_entry(self):
        level = self.config.level
        family = random.choice(["weekday", "interval", "monthend"])
        n = max(3, 2 + level * 6 + random.randrange(0, 8))

        if family == "weekday":
            K = random.choice([1, 2, 3, 4, 6])
            if random.choice(["nth", "last"]) == "nth":
                o = random.randint(1, 4)
                wd = random.randint(0, 4)
                spec = ("nth", o, wd)
                basis = "the {} {} of {}".format(_ordinal(o),
                                                 WEEKDAY_NAMES[wd],
                                                 _fmt_months(K))
            else:
                wd = random.randint(0, 4)
                spec = ("last", wd)
                basis = "the last {} of {}".format(WEEKDAY_NAMES[wd],
                                                   _fmt_months(K))
            start_y = random.randint(2015, 2030)
            start_m = random.randint(1, 12)
            dates = []
            y, m = start_y, start_m
            for _ in range(n):
                dates.append(_month_date(y, m, spec))
                y, m = _add_months(y, m, K)
            when = "starting in {} {}".format(MONTH_NAMES[start_m - 1],
                                              start_y)
            skip = ""
            extra = dict(family=family, K=K, spec=tuple(spec),
                         year=start_y, month=start_m)

        elif family == "interval":
            D = random.randint(15, 90)
            start = date(random.randint(2015, 2030),
                         random.randint(1, 12), random.randint(1, 28))
            dates = [_business(start + timedelta(days=i * D))
                     for i in range(n)]
            basis = "every {} days".format(D)
            when = "from {}".format(start.isoformat())
            skip = (", skipping any occurrence that falls on a weekend by "
                    "shifting it to the following Monday")
            extra = dict(family=family, D=D, start_iso=start.isoformat())

        else:
            K = random.choice([1, 2, 3, 4, 6, 12])
            kind = random.choice(["day", "weekday"])
            start_y = random.randint(2015, 2030)
            start_m = random.randint(1, 12)
            dates = []
            last_spec = None
            y, m = start_y, start_m
            for _ in range(n):
                if kind == "day":
                    last_spec = ("lastday",)
                else:
                    wd = random.randint(0, 4)
                    last_spec = ("last", wd)
                dates.append(_month_date(y, m, last_spec))
                y, m = _add_months(y, m, K)
            basis = "the last day" if kind == "day" else "the last weekday"
            when = "starting in {} {}".format(MONTH_NAMES[start_m - 1],
                                              start_y)
            skip = ""
            extra = dict(family=family, K=K, kind=kind,
                         year=start_y, month=start_m, spec=tuple(last_spec))

        answer = dates[n - 1]
        clause = "{}, {}{}.".format(basis, when, skip)
        cot = ("Iterated the recurrence occurrence by occurrence with "
               "datetime.date arithmetic (shifting weekend dates to the "
               "following Monday where the rule requires it) and took the "
               "n-th entry.")

        metadata = edict(family=family, n=n, clause=clause, cot=cot)
        metadata.payload = {"recurrence": clause}
        metadata.update(extra)
        return Entry(metadata=metadata, answer=answer.isoformat())

    def render_prompt(self, metadata):
        return ("{} What is the date (YYYY-MM-DD) of the {} occurrence? "
                "The answer is a date in the format YYYY-MM-DD.".format(
                    metadata.clause, _ordinal(metadata.n)))

    def score_answer(self, answer, entry):
        return _score(answer, entry)


def _score(answer, entry):
    try:
        return 1.0 if str(answer).strip() == entry.answer else 0.0
    except Exception:
        return 0.0
