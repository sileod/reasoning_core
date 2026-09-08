"""Semantic paraphrase: rewrite a generated event under a requested construction.

Given a source sentence describing a simple event (subject-verb-object with an
optional negation), the model must rewrite it under a requested construction
(active/passive voice, present/past tense, affirmative/negated polarity, or any
combination) while preserving the participants, the non-flipped features and the
overall meaning. The canonical answer is the exactly determined transformed
sentence, which varies widely across examples (low constant-guess rate).
"""

import random

from reasoning_core.template import Task, Entry, Config, edict

AGENTS = [
    "chef", "cat", "poet", "sailor", "child", "wolf",
    "builder", "artist", "doctor", "farmer", "captain", "cook",
]

PATIENTS = [
    "meal", "mouse", "letter", "ship", "toy", "sheep",
    "house", "portrait", "patient", "crop", "parcel", "report",
]

# base, present-3sg, past, past-participle
VERBS = {
    "eat": ("eat", "eats", "ate", "eaten"),
    "take": ("take", "takes", "took", "taken"),
    "write": ("write", "writes", "wrote", "written"),
    "give": ("give", "gives", "gave", "given"),
    "build": ("build", "builds", "built", "built"),
    "send": ("send", "sends", "sent", "sent"),
    "draw": ("draw", "draws", "drew", "drawn"),
    "paint": ("paint", "paints", "painted", "painted"),
}


def _render(agent, patient, verb, voice, polarity, tense):
    base, pres3, past, pp = VERBS[verb]
    a = "the " + agent
    p = "the " + patient
    be = "is" if tense == "present" else "was"
    if voice == "active":
        if tense == "present" and polarity == "affirmative":
            return f"{a} {pres3} {p}."
        if tense == "present" and polarity == "negated":
            return f"{a} does not {base} {p}."
        if tense == "past" and polarity == "affirmative":
            return f"{a} {past} {p}."
        return f"{a} did not {base} {p}."
    if polarity == "affirmative":
        return f"{p} {be} {pp} by {a}."
    return f"{p} {be} not {pp} by {a}."


def _construction(features):
    parts = []
    if "voice" in features:
        parts.append("in the active voice" if features["voice"] == "active" else "in the passive voice")
    if "polarity" in features:
        parts.append("without negation" if features["polarity"] == "affirmative" else "with a negation")
    if "tense" in features:
        parts.append("in the present tense" if features["tense"] == "present" else "in the past tense")
    return ", and ".join(parts)


def _norm(s):
    return " ".join(str(s).lower().replace(".", "").split())


class ParaphraseConfig(Config):
    max_flips: int = 1

    def apply_difficulty(self, level):
        self.max_flips = min(3, 1 + (level + 1) // 2)


class SemanticParaphrase(Task):
    summary = ("Express a generated event using a requested construction while preserving "
               "participants, negation, tense, and meaning; modes flip voice (active/passive), "
               "polarity (affirmative/negated), and tense (present/past), singly or combined.")
    config_cls = ParaphraseConfig

    def generate_entry(self):
        dims = ["voice", "polarity", "tense"]
        flips = random.sample(dims, k=random.randint(1, self.config.max_flips))

        agent = random.choice(AGENTS)
        patient = random.choice(PATIENTS)
        verb = random.choice(list(VERBS.keys()))

        source = {d: random.choice(["active", "passive"]) if d == "voice"
                  else random.choice(["affirmative", "negated"]) if d == "polarity"
                  else random.choice(["present", "past"])
                  for d in dims}

        target = dict(source)
        for d in flips:
            if d == "voice":
                target[d] = "passive" if source[d] == "active" else "active"
            elif d == "polarity":
                target[d] = "negated" if source[d] == "affirmative" else "affirmative"
            else:
                target[d] = "past" if source[d] == "present" else "present"

        source_text = _render(agent, patient, verb, source["voice"], source["polarity"], source["tense"])
        answer = _render(agent, patient, verb, target["voice"], target["polarity"], target["tense"])
        construction = _construction({d: target[d] for d in flips})

        payload = {"source": source_text, "construction": construction}
        metadata = edict({
            "agent": agent,
            "patient": patient,
            "verb": verb,
            "source": {k: str(v) for k, v in source.items()},
            "target": {k: str(v) for k, v in target.items()},
            "flips": flips,
            "construction": construction,
            "payload": payload,
        })

        assert _render(agent, patient, verb, target["voice"], target["polarity"], target["tense"]) == answer

        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (
            f"The following sentence describes an event: {metadata.payload['source']}\n"
            f"Rewrite this sentence so that it is {metadata.payload['construction']}, keeping the "
            f"same participants and overall meaning and leaving every other feature (voice, polarity, "
            f"tense) as it currently is unless the construction changes it.\n"
            f"The answer is the rewritten sentence."
        )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str) or not answer.strip():
            return 0.0
        return 1.0 if _norm(answer) == _norm(entry.answer) else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'semantic_paraphrase (draw 1 of 2)',
 'hypothesis': 'ASTRA0-10',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave10/semantic_paraphrase',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 836983008,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
