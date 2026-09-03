import random
from dataclasses import dataclass
from datetime import date

from reasoning_core.template import Task, Entry, Config, edict

TASK_META = {'parent_source_id': None,
 'idea': 'gregorian_weekday (draw 1 of 2)',
 'hypothesis': 'W1-075',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/gregorian_weekday',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 4114080681,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass
class GregorianWeekdayConfig(Config):
    min_year: int = 1800
    max_year: int = 2199

    def apply_difficulty(self, level):
        self.min_year = int(self.min_year) - level * 150
        self.max_year = int(self.max_year) + level * 150


def _pick_year(config):
    while True:
        year = random.randint(config.min_year, config.max_year)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        try:
            date(year, month, day)
            return year, month, day
        except ValueError:
            pass


class GregorianWeekday(Task):
    summary = ("Given a valid Gregorian date (year month day), output its weekday as "
               "one of Monday..Sunday; the date ranges across a wide span of years so "
               "century/leap-year shift rules matter.")
    config_cls = GregorianWeekdayConfig

    def generate_entry(self):
        year, month, day = _pick_year(self.config)
        d = date(year, month, day)
        answer = WEEKDAYS[d.weekday()]
        metadata = edict({
            "year": year,
            "month": month,
            "day": day,
        })
        metadata.payload = {"year": year, "month": month, "day": day}
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (f"Using the standard Gregorian calendar, what weekday is "
                f"{metadata.day} {metadata.month} {metadata.year}? "
                f"The answer is the weekday name, one of Monday, Tuesday, Wednesday, "
                f"Thursday, Friday, Saturday, Sunday.")

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        return 1.0 if answer.strip() == entry.answer else 0.0
