"""question_generation: generate the question whose answer is a marked constituent."""

import random

from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload

TASK_META = {'parent_source_id': None,
 'idea': 'question_generation (draw 1 of 2)',
 'hypothesis': 'ASTRA0-09',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave10/question_generation',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3704315384,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

ROLES = ("subj", "obj", "time", "place", "reason", "manner")

SUBJ_PERSON = [
    ("the chef", "cook", "cooked"), ("the gardener", "plant", "planted"),
    ("the mechanic", "fix", "fixed"), ("the teacher", "mark", "marked"),
    ("the artist", "paint", "painted"), ("the pilot", "fly", "flew"),
]
SUBJ_THING = [
    ("the storm", "shake", "shook"), ("the engine", "power", "powered"),
    ("the river", "carve", "carved"), ("the magnet", "draw", "drew"),
    ("the algorithm", "process", "processed"),
]
OBJ_PERSON = ["the visitor", "the student", "the guest", "the customer"]
OBJ_THING = ["the parcel", "the report", "the bridge", "the orchard", "the lantern"]
TIME = ["at dawn", "at midnight", "after sunset", "before noon", "during the festival"]
PLACE = ["in the harbor", "by the coast", "behind the mill", "near the plaza"]
REASON = ["because it was overdue", "to honor the tradition", "to test the design",
          "on account of the storm"]
MANNER = ["carefully", "swiftly", "reluctantly", "with great precision"]
SPEC = {
    "time": TIME, "place": PLACE, "reason": REASON, "manner": MANNER,
}


def _pick(pool):
    return pool[random.randrange(len(pool))]


def _whword(which, subj_class, obj_class):
    if which == "subj":
        return "who" if subj_class == "person" else "what"
    if which == "obj":
        return "whom" if obj_class == "person" else "what"
    return {"time": "when", "place": "where", "reason": "why", "manner": "how"}[which]


def _gen(which, subj, vbase, vpast, obj, ctx, subj_class, obj_class):
    """Return (sentence_with_bracket, question)."""
    wh = _whword(which, subj_class, obj_class)
    aux = "did"

    segs = {}
    segs["subj"] = subj
    segs["obj"] = obj
    segs.update(ctx)

    def decl_order():
        for r in ("subj", "verb", "obj", "time", "place", "reason", "manner"):
            if r == "verb":
                yield vpast
            else:
                val = segs.get(r)
                if val is None:
                    continue
                yield ("[" + val + "]") if which == r else val

    sentence = " ".join(t for t in decl_order() if t) + "."
    sentence = sentence[0].upper() + sentence[1:]

    if which == "subj":
        parts = []
        for r in ("verb", "obj", "time", "place", "reason", "manner"):
            v = segs.get(r)
            if r == "verb":
                v = vpast
            if v:
                parts.append(v)
        question = wh.capitalize() + " " + " ".join(parts) + "?"
    else:
        parts = []
        for r in ("obj", "time", "place", "reason", "manner"):
            if r == which:
                continue
            v = segs.get(r)
            if v:
                parts.append(v)
        question = (wh.capitalize() + " " + aux + " " + subj + " " + vbase +
                    ((" " + " ".join(parts)) if parts else "") + "?")
    question = question[0].upper() + question[1:]
    return sentence, question


def _normalize(text):
    return " ".join(str(text).strip().lower().replace("\n", " ").split())


@dataclass
class QuestionGenerationV1Config(Config):
    n_context: int = 2

    def apply_difficulty(self, level):
        self.n_context = 2 + level
        return self


class QuestionGeneration(Task):
    summary = ("Generate the wh-question answered by a bracketed constituent of a "
               "declarative sentence across subject, object, time, place, reason and "
               "manner roles in one-context and multi-context sentences.")
    config_cls = QuestionGenerationV1Config

    def generate_entry(self):
        if random.random() < 0.5:
            subj, vbase, vpast = _pick(SUBJ_PERSON)
            subj_class = "person"
        else:
            subj, vbase, vpast = _pick(SUBJ_THING)
            subj_class = "thing"
        obj = _pick(OBJ_PERSON if random.random() < 0.5 else OBJ_THING)
        obj_class = "person" if obj in OBJ_PERSON else "thing"

        ctx = {}
        n_ctx = self.config.n_context
        avail = ["time", "place", "reason", "manner"]
        random.shuffle(avail)
        for r in avail[:n_ctx]:
            ctx[r] = _pick(SPEC[r])

        which = random.choice(list(ctx.keys()) + ["subj", "obj"])
        sentence, question = _gen(which, subj, vbase, vpast, obj, ctx, subj_class, obj_class)

        assert "[" in sentence and "]" in sentence
        metadata = edict({
            "sentence": sentence,
            "constituent_role": which,
            "question": question,
            "payload": {
                "sentence": sentence,
                "prompt": (
                    "The bracketed fragment [ ... ] inside the sentence below is the "
                    "answer to a wh-question. Write that question. Your answer is "
                    "exactly the question, nothing else."
                ),
            },
        })
        return Entry(metadata=metadata, answer=question)

    def render_prompt(self, metadata):
        return (
            f"{render_payload(metadata.payload)}\n\n"
            f"{metadata.sentence}\n\n"
            f"Write the wh-question whose answer is the bracketed constituent [ ... ]. "
            f"Your answer is exactly that question."
        )

    def score_answer(self, answer, entry):
        if not answer or not isinstance(answer, str):
            return 0.0
        return 1.0 if _normalize(answer) == _normalize(entry.answer) else 0.0
