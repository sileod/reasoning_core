import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround


@dataclass
class EventOrderConfig(Config):
    n_events: int = 4
    max_flashbacks: int = 2

    def apply_difficulty(self, level):
        self.n_events = sround(4 + level * 2)
        self.max_flashbacks = sround(1 + level * 0.8)


def _event(ev, labels):
    return labels[ev]


class EventOrderReconstruction(Task):
    summary = "Recover chronological event order from a short narrative containing flashbacks and temporal connectives."

    config_cls = EventOrderConfig

    def generate_entry(self):
        n = self.config.n_events
        chronological_seq = list(range(n))
        random.shuffle(chronological_seq)

        words = ["breakfast", "walk", "meeting", "lunch", "call", "nap",
                 "dinner", "run", "read", "shop", "cook", "write",
                 "email", "bike", "swim", "sketch", "sing", "garden",
                 "paint", "hike", "bake", "dance", "stretch", "meditate"]
        labels = sorted(random.sample(words, n))

        narrative_parts = []
        seen = []
        for idx, ev in enumerate(chronological_seq):
            if random.random() < 0.5 and seen:
                fb = random.choice(seen)
                narrative_parts.append(
                    "the %s actually happened earlier, before the %s, even though it was told now" % (_event(fb, labels), _event(ev, labels)))
                narrative_parts.append("the %s then occurred" % _event(ev, labels))
            else:
                seqword = {0: "first,", 1: "next,"}.get(idx, "after that,")
                if idx == n - 1:
                    seqword = "finally,"
                narrative_parts.append("%s the %s happened" % (seqword, _event(ev, labels)))
            seen.append(ev)

        answer = ",".join(labels[i] for i in chronological_seq)

        metadata = edict({
            "n_events": n,
            "labels": labels,
            "chronological_seq": chronological_seq,
            "answer": answer,
        })
        metadata.payload = {
            "narrative": " ".join(narrative_parts) + ".",
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        narrative = metadata.payload["narrative"]
        return (
            "Read this narrative: " + narrative + "\n"
            "The narrative may tell events out of true order because some were recalled "
            "as flashbacks. List the events in the true chronological order, earliest first, "
            "separated by commas. Use the exact event names.\n"
            "The answer is a comma-separated list of the events in chronological order."
        )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        got = ",".join(a.strip() for a in answer.replace(" and ", ",").split(",") if a.strip())
        return 1.0 if got == entry.answer else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'event_order_reconstruction (draw 1 of 2)',
 'hypothesis': 'ASTRA0-17',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave10/event_order_reconstruction',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 69737855,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
