import random
from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, edict


@dataclass
class TemporalPerspectiveV1Config(Config):
    n_events: int = 3
    max_span: int = 10
    time_units: str = "hours"

    def apply_difficulty(self, level):
        self.n_events = 3 + level
        self.max_span = 10 + 4 * level
        self.time_units = "hours"


class TemporalPerspective(Task):
    summary = (
        "Express an event's status from a specified reference time, "
        "distinguishing ongoing, completed, and not yet started."
    )
    config_cls = TemporalPerspectiveV1Config

    def generate_entry(self):
        cfg = self.config
        n = cfg.n_events
        max_span = cfg.max_span

        while True:
            starts = [random.randint(0, max_span) for _ in range(n)]
            lasts = [random.randint(1, max_span) for _ in range(n)]
            ref = random.randint(0, max_span + 4)
            events = []
            for i in range(n):
                s = starts[i]
                d = lasts[i]
                if ref < s:
                    status = "not_yet_started"
                    val = s - ref
                elif ref < s + d:
                    status = "ongoing"
                    val = ref - s
                else:
                    status = "completed"
                    val = ref - (s + d)
                events.append({
                    "name": chr(ord("A") + i),
                    "start": s,
                    "duration": d,
                    "status": status,
                    "val": val,
                    "start_str": f"{s} {cfg.time_units} into the day",
                    "duration_str": f"{d} {cfg.time_units}",
                })

            statuses = sorted([e["status"] for e in events])
            if len(set(statuses)) < 2:
                continue

            answer = "\n".join(
                f"{e['name']}: {e['status']}" +
                (f"; {e['val']} {cfg.time_units} " +
                 ("into the event" if e["status"] == "ongoing"
                  else ("until start" if e["status"] == "not_yet_started"
                        else "since end")))
                for e in events
            )

            payload = {
                "reference_time": f"At {ref} {cfg.time_units} into the day, "
                                  f"classify each event below.",
                "events": [
                    {
                        "name": e["name"],
                        "start": e["start_str"],
                        "duration": e["duration_str"],
                    }
                    for e in events
                ],
            }
            metadata = edict({
                "starts": starts,
                "lasts": lasts,
                "ref": ref,
                "events": [
                    {
                        "name": e["name"],
                        "start": e["start"],
                        "duration": e["duration"],
                        "status": e["status"],
                        "val": e["val"],
                    }
                    for e in events
                ],
            })
            metadata.payload = payload
            metadata.time_units = cfg.time_units
            return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = [
            metadata.payload["reference_time"],
            "",
        ]
        for ev in metadata.payload["events"]:
            lines.append(
                f"- {ev['name']}: starts {ev['start']}, lasts {ev['duration']}."
            )
        lines.append("")
        lines.append(
            "For each event, write a line `NAME: STATUS` where STATUS is "
            "`ongoing`, `completed`, or `not_yet_started`. An event is "
            "`completed` if the reference time is past start+duration, "
            "`ongoing` if the reference time is within [start, start+duration), "
            "and `not_yet_started` if it is before start."
        )
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        expected = entry.answer
        return float(answer.strip() == expected.strip())


TASK_META = {'parent_source_id': None,
 'idea': 'temporal_perspective (draw 1 of 2)',
 'hypothesis': 'ASTRA2-temporal_perspective',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave11/temporal_perspective',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 4275654395,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
