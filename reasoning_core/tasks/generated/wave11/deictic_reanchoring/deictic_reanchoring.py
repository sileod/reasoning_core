"""Deictic re-anchoring: rewrite 'I', 'here', 'tomorrow' and similar expressions
after changes of speaker, place, or time.

A speaker makes a direct utterance that is dense in indexical expressions. The
statement must be transcribed at a later date, in a new location, to a new
audience, where those indexicals would no longer resolve -- so every indexical is
expanded to its explicit referent (a person's name, a place's name, a concrete
day). The expansion is computed from a fully-stated people/place/day schedule so
the answer is deterministic and exactly verifiable.
"""
import random
import re
from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, edict

PEOPLE = ("Alice", "Bob", "Carol", "Dave", "Eve", "Frank",
          "Grace", "Hank", "Iris", "Jack")
PLACES = ("the garden", "the office", "the market", "the kitchen",
          "the beach", "the park", "the library", "the station",
          "the bakery", "the museum", "the farm", "the theatre",
          "the gym", "the school")

# name_of_day -> day index (0-based); day labels are "Day 1".."Day T"
# Each template is (source_text, clauses) where clauses is a list of
# (who, where_attr, when) describing how to build the re-anchored clause.
# who in {S, A}; where_attr in {"here", "there"}; when in {"today","yesterday","tomorrow"}


def _reanchor_clauses(template_id, s_name, a_name, schedule, s_idx, a_idx, d):
    """Return list of re-anchored clause strings for a template.

    T = d (today's index 1..T). All indexes returned are 0-based day.
    """
    here_day = {"today": d, "tomorrow": d + 1, "yesterday": d - 1}

    def place(who, attr, when):
        day = here_day[when]
        idx = s_idx if who == "S" else a_idx
        return schedule[idx][day]

    def day_label(day):
        return f"Day {day + 1}"

    def clause(who, attr, when, tense):
        p = place(who, attr, when)
        dl = day_label(here_day[when])
        name = s_name if who == "S" else a_name
        return f"{name} {tense} at {p} on {dl}"

    if template_id == 1:
        # "I am here today, and you are there."
        c1 = clause("S", "here", "today", "is")
        c2 = clause("A", "there", "today", "is")
        return [c1, c2]
    if template_id == 2:
        # "I will be here tomorrow, and you will be there tomorrow."
        return [clause("S", "here", "tomorrow", "will be"),
                clause("A", "there", "tomorrow", "will be")]
    if template_id == 3:
        # "I was here yesterday, and you were there yesterday."
        return [clause("S", "here", "yesterday", "was"),
                clause("A", "there", "yesterday", "were")]
    if template_id == 4:
        # "I am here today."
        return [clause("S", "here", "today", "is")]
    if template_id == 5:
        # "I am here today, and I will be here tomorrow."
        return [clause("S", "here", "today", "is"),
                clause("S", "here", "tomorrow", "will be")]
    if template_id == 6:
        # "I was here yesterday, and I am here today."
        return [clause("S", "here", "yesterday", "was"),
                clause("S", "here", "today", "is")]
    raise ValueError(template_id)


SOURCE_TEXTS = {
    1: "I am here today, and you are there.",
    2: "I will be here tomorrow, and you will be there tomorrow.",
    3: "I was here yesterday, and you were there yesterday.",
    4: "I am here today.",
    5: "I am here today, and I will be here tomorrow.",
    6: "I was here yesterday, and I am here today.",
}

TEMPLATE_DAYS = {1: 0, 2: 1, 3: -1, 4: 0, 5: 1, 6: -1}  # max offset needed (1->+1, -1->minus)


def _normalize(text):
    text = str(text or "").strip()
    text = text.rstrip(".")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


class DeicticReanchoringConfig(Config):
    n_people: int = 2
    n_days: int = 4
    use_addressee: bool = True
    use_temporal: bool = False
    max_attempts: int = 200

    def apply_difficulty(self, level):
        # monotonic: more people, longer timeline, addressee & relative-day terms.
        self.n_people = 2 + (1 if level >= 3 else 0) + (1 if level >= 5 else 0)
        self.n_days = min(3 + level, 8)
        self.use_addressee = True
        self.use_temporal = level >= 2
        self.max_attempts = 200 + 50 * level


class DeicticReanchoring(Task):
    """Track and expand deictic expressions after shifts of speaker, place, and time."""

    summary = ("Rewrite 'I', 'here', 'tomorrow', and similar expressions after "
               "changes of speaker, place, or time.")
    config_cls = DeicticReanchoringConfig
    task_version = 2

    def _template_pool(self):
        base = [1, 4]
        if self.config.use_addressee:
            base = [1, 2, 3, 4, 5, 6]
        return base

    def _make_schedule(self, m, t):
        attempt = 0
        while attempt < 200:
            attempt += 1
            schedule = []
            for _ in range(m):
                row = [random.choice(PLACES) for _ in range(t)]
                schedule.append(row)
            yield schedule

    def _build(self):
        cfg = self.config
        m = cfg.n_people
        t = cfg.n_days
        names = random.sample(PEOPLE, m)
        templates = self._template_pool()
        tid = random.choice(templates)
        need_days = TEMPLATE_DAYS[tid]
        # choose day d (1..t 1-based) such that d+need in [1,t]
        lo = 1 if need_days >= 0 else -need_days + 1
        hi = t if need_days <= 0 else t - need_days
        if hi < lo:
            return None
        d = random.randint(lo, hi)  # 1-based index into rows (rows are 0-based)
        d0 = d - 1
        s_idx = random.randrange(m)
        candidates = [i for i in range(m) if i != s_idx]
        if not candidates:
            return None
        a_idx = random.choice(candidates)

        for schedule in self._make_schedule(m, t):
            # check there/here distinctness where needed
            ok = True
            if a_idx is not None:
                for when in ("today", "tomorrow", "yesterday"):
                    if when not in ("today", "tomorrow", "yesterday"):
                        continue
                    if tid in (1,) and when != "today":
                        continue
                    if tid == 2 and when != "tomorrow":
                        continue
                    if tid == 3 and when != "yesterday":
                        continue
                    if tid in (1, 2, 3):
                        day = {"today": d0, "tomorrow": d0 + 1,
                               "yesterday": d0 - 1}[when]
                        if schedule[s_idx][day] == schedule[a_idx][day]:
                            ok = False
                            break
            if not ok:
                continue

            s_name = names[s_idx]
            a_name = names[a_idx] if a_idx is not None else s_name
            clauses = _reanchor_clauses(tid, s_name, a_name, schedule,
                                        s_idx, a_idx if a_idx is not None else s_idx, d0)
            answer = "; ".join(clauses)
            return (tid, names, schedule, d, s_idx, a_idx, s_name, a_name,
                    clauses, answer)
        return None

    def generate_entry(self):
        for _ in range(self.config.max_attempts):
            built = self._build()
            if built is None:
                continue
            (tid, names, schedule, d, s_idx, a_idx, s_name, a_name,
             clauses, answer) = built
            t = self.config.n_days

            # Verify the gold: reconstruct the intended referents by mapping each
            # clause back onto the schedule and confirming it is reproducible.
            if not _verify_answer(tid, s_name, a_name, schedule, s_idx, a_idx, d, clauses):
                continue

            sched_lines = []
            for day in range(t):
                places = ", ".join(
                    f"{names[p]} is at {schedule[p][day]}" for p in range(len(names))
                )
                sched_lines.append(f"Day {day + 1}: {places}.")
            schedule_text = "\n".join(sched_lines)

            src = SOURCE_TEXTS[tid]
            where_here = schedule[s_idx][d - 1]
            meta = edict({
                "people": names,
                "schedule": {names[p]: [schedule[p][i] for i in range(t)]
                             for p in range(len(names))},
                "n_days": t,
                "speaker": s_name,
                "addressee": a_name,
                "speaking_day": d,
                "speaker_here": where_here,
                "template_id": tid,
                "source_utterance": src,
                "schedule_text": schedule_text,
                "answer_clauses": clauses,
                "payload": {
                    "schedule": schedule_text,
                    "speaker": s_name,
                    "addressee": a_name,
                    "speaking_day": f"Day {d}",
                    "source": src,
                },
            })
            return Entry(metadata=meta, answer=answer)
        raise RuntimeError(
            f"Could not build a deictic re-anchoring instance after "
            f"{self.config.max_attempts} attempts (n_people={self.config.n_people}, "
            f"n_days={self.config.n_days})")

    def render_prompt(self, metadata):
        days = {str(i + 1): f"Day {i + 1}" for i in range(metadata["n_days"])}
        return (
            "A statement is later transcribed in a different place at a later "
            "time, to a new audience, where the speaker's indexical words would "
            "no longer be clear. Replace every indexical with its explicit "
            "referent. In the direct utterance, 'I' is the speaker NAME, 'you' "
            "is the addressee NAME, 'here' is the named place from the relevant "
            "day of the schedule, 'there' is the addressee's named place that "
            "day, and 'today'/'yesterday'/'tomorrow' are the concrete Day N of "
            "the schedule. Report each clause as 'NAME is/was/will be at PLACE "
            "on Day N', separated by ' ; '.\n\n"
            f"{metadata['payload']['schedule']}\n\n"
            f"On {metadata['payload']['speaking_day']}, {metadata['payload']['speaker']} "
            f"told {metadata['payload']['addressee']}:\n"
            f"\"{metadata['payload']['source']}\"\n\n"
            "Your transcription of that statement is:"
        )

    def score_answer(self, answer, entry):
        return float(_normalize(answer) == _normalize(entry.answer))


def _verify_answer(tid, s_name, a_name, schedule, s_idx, a_idx, d, clauses):
    """Confirm the re-anchored clauses reproduce the intended referents.

    Re-derives each clause's place/day from the schedule and cross-checks against
    the built clause strings. Returns False (reject) if any mismatch.
    """
    here_day = {"today": d - 1, "tomorrow": d, "yesterday": d - 2}
    want = []
    src = SOURCE_TEXTS[tid]
    # Determine the clause structure from the template
    if tid in (1, 2, 3):
        who_seq = ["S", "A"]
        attr_seq = ["here", "there"]
        when = {"1": "today", "2": "tomorrow", "3": "yesterday"}[str(tid)]
        for who, attr in zip(who_seq, attr_seq):
            idx = s_idx if who == "S" else a_idx
            day = here_day[when]
            tense = {"today": "is", "tomorrow": "will be", "yesterday": "was"}[when]
            if tid == 3 and who == "A":
                tense = "were"
            name = s_name if who == "S" else a_name
            want.append(f"{name} {tense} at {schedule[idx][day]} on Day {day + 1}")
    elif tid == 4:
        day = here_day["today"]
        want.append(f"{s_name} is at {schedule[s_idx][day]} on Day {day + 1}")
    elif tid == 5:
        d1 = here_day["today"]
        d2 = here_day["tomorrow"]
        want.append(f"{s_name} is at {schedule[s_idx][d1]} on Day {d1 + 1}")
        want.append(f"{s_name} will be at {schedule[s_idx][d2]} on Day {d2 + 1}")
    elif tid == 6:
        d1 = here_day["yesterday"]
        d2 = here_day["today"]
        want.append(f"{s_name} was at {schedule[s_idx][d1]} on Day {d1 + 1}")
        want.append(f"{s_name} is at {schedule[s_idx][d2]} on Day {d2 + 1}")
    else:
        return False
    return [c for c in clauses] == want


TASK_META = {'parent_source_id': None,
 'idea': 'deictic_reanchoring (draw 2 of 2)',
 'hypothesis': 'ASTRA2-deictic_reanchoring',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave11/deictic_reanchoring',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 360709197,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
