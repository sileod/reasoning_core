import random
from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, edict, render_payload, stochastic_rounding as sround

NAMES = [
    "Mira", "Theo", "Iris", "Omar", "Nina", "Leo", "Zara", "Kai",
    "Eli", "Rhea", "Jude", "Anya", "Finn", "Lara", "Milo", "Sana",
    "Hugo", "Tess", "Ravi", "Cleo", "Bram", "Yara", "Ezra", "Ivo",
]


@dataclass
class LosslessSummaryConfig(Config):
    n_queried: int = 3
    n_distract: int = 3
    score_lo: int = 5
    score_hi: int = 48
    year_lo: int = 2001
    year_hi: int = 2024

    def apply_difficulty(self, level):
        self.n_queried = sround(2 + level)
        self.n_distract = sround(2 + level * 2)
        self.score_hi = min(48 + level * 4, 98)


def _build(config):
    total = config.n_queried + config.n_distract
    names = random.sample(NAMES, total)
    scores = random.sample(range(config.score_lo, config.score_hi), total)
    score_map = dict(zip(names, scores))

    queried = names[: config.n_queried]
    distract = names[config.n_queried:]
    random.shuffle(queried)

    entries = list(zip(names, scores))
    random.shuffle(entries)

    clauses = []
    for i, (name, score) in enumerate(entries):
        if i == 0:
            clauses.append(f"{name} scored {score} points")
        else:
            clauses.append(f"{name} scored {score} points")
    year = random.randint(config.year_lo, config.year_hi)
    room = random.randint(101, 499)
    narrative = (
        "A mixed relay round just ended. "
        + ", and ".join(clauses)
        + f". Every score is an exact, distinct whole number. The event kicked off around "
        + f"{year} and took place in room {room}, but neither the year nor the room "
        + "affects any score."
    )

    query_order = ", ".join(queried)
    answer = ", ".join(str(score_map[q]) for q in queried)
    return {
        "names": names,
        "scores": scores,
        "queried": queried,
        "distract": distract,
        "score_map": score_map,
        "narrative": narrative,
        "query_order": query_order,
        "answer": answer,
    }


class LosslessSummary(Task):
    summary = (
        "Compress a short generated narrative while preserving exactly the facts "
        "needed for specified future queries: pick the queried participants' scores "
        "out of a jumbled paragraph of scores, years and room numbers, and emit just "
        "those values in the listed order."
    )
    config_cls = LosslessSummaryConfig

    def generate_entry(self):
        data = _build(self.config)
        metadata = edict({"narrative": data["narrative"], "query_order": data["query_order"]})
        metadata.payload = {
            "narrative": data["narrative"],
            "query_order": data["query_order"],
            "answer": data["answer"],
        }
        assert all(isinstance(s, int) and s >= 0 for s in data["scores"])
        return Entry(metadata=metadata, answer=data["answer"])

    def render_prompt(self, metadata):
        return (
            f"{metadata.narrative}\n\n"
            f"A future query will ask: report the final score of each of these "
            f"participants, in this exact order: {metadata.query_order}.\n\n"
            "Write a lossless summary that keeps exactly the facts needed for that "
            "query. Give the scores in the order the participants were listed, "
            "separated by commas and nothing else. For example, if the three asked "
            "scores were 5, 9 and 2 the answer format is \"5, 9, 2\".\n\n"
            "Answer:"
        )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        return 1.0 if answer.strip() == entry.answer else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'lossless_summary (draw 1 of 2)',
 'hypothesis': 'ASTRA2-lossless_summary',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave11/lossless_summary',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3797320393,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
