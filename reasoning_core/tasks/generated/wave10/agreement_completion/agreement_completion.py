"""Subject-verb agreement completion task.

Given a sentence fragment whose verb has been replaced by a blank, pick the
correctly inflected present-tense form of a stated verb. The difficulty lies in
locating the grammatical subject: intervening prepositional phrases, relative
clauses with invariant auxiliaries, and coordinated noun phrases all sit between
the subject head and the blank, and only the head (or the conjunction, for
compound subjects) determines the number the verb must agree with.
"""

import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict

TASK_META = {'parent_source_id': None,
 'idea': 'agreement_completion (draw 2 of 2)',
 'hypothesis': 'ASTRA0-11',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave10/agreement_completion',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2738521636,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

# Singular and plural noun head forms. Only regular plurals are used so the
# head noun's number is always unambiguous from its surface form.
SING = (
    "cat", "dog", "bird", "tree", "car", "book", "box", "bus", "city",
    "story", "fox", "dish", "hero", "potato", "wolf", "leaf", "child",
    "mouse", "foot", "woman",
)
PLUR = (
    "cats", "dogs", "birds", "trees", "cars", "books", "boxes", "buses",
    "cities", "stories", "foxes", "dishes", "heroes", "potatoes", "wolves",
    "leaves", "children", "mice", "feet", "women",
)

# (base, third-person-singular). Verb bank restricted to forms that can be used
# intransitively with a locative/temporal adverb so the finished sentence stays
# grammatical for every subject.
VERBS = (
    ("run", "runs"), ("walk", "walks"), ("fly", "flies"), ("bark", "barks"),
    ("sing", "sings"), ("sleep", "sleeps"), ("swim", "swims"), ("jump", "jumps"),
    ("sit", "sits"), ("stand", "stands"), ("dance", "dances"), ("cry", "cries"),
    ("howl", "howls"), ("gallop", "gallops"), ("crawl", "crawls"), ("drift", "drifts"),
    ("sail", "sails"), ("soar", "soars"), ("glide", "glides"), ("ring", "rings"),
    ("grow", "grows"), ("bloom", "blooms"), ("roam", "roams"), ("wander", "wanders"),
    ("vanish", "vanishes"), ("hurry", "hurries"), ("live", "lives"), ("play", "plays"),
    ("work", "works"), ("laugh", "laughs"), ("glow", "glows"), ("shine", "shines"),
    ("float", "floats"), ("twitch", "twitches"), ("shiver", "shivers"),
    ("whisper", "whispers"), ("screech", "screeches"), ("patter", "patters"),
    ("scamper", "scampers"), ("twitter", "twitters"),
)

PREPS = ("of", "with", "for", "from", "in", "near", "beside")
DETS_SING = ("a", "the")
DETS_PLUR = ("the", "several", "many")
MODAL = ("can", "will", "must", "should", "could", "might", "may")
COMPLEMENTS = (
    "loudly", "at dawn", "in the meadow", "near the river", "every morning",
    "in the pond", "under the tree", "at night", "in the forest", "all day",
)


def _rand_noun():
    """A noun with a number chosen to mismatch the head and lure the model."""
    pool = (SING if random.random() < 0.5 else PLUR)
    return random.choice(pool)


def _modifier_block():
    """One distractor inserted between the subject head and the verb."""
    if random.random() < 0.5:
        prep = random.choice(PREPS)
        if random.random() < 0.6:
            det = random.choice(("a", "the", "several", "many"))
        else:
            det = random.choice(("a", "the"))
        noun = _rand_noun()
        return "%s %s %s" % (prep, det, noun)
    modal = random.choice(MODAL)
    verb = random.choice([base for base, _ in VERBS])
    if random.random() < 0.5:
        return "that %s %s" % (modal, verb)
    prep = random.choice(PREPS)
    noun = _rand_noun()
    return "that %s %s %s the %s" % (modal, verb, prep, noun)


@dataclass
class AgreementCompletionV2Config(Config):
    n_mods: int = 1
    compound_prob: float = 0.08

    def apply_difficulty(self, level):
        self.n_mods = 1 + level
        self.compound_prob = min(0.08 + 0.05 * level, 0.4)


class AgreementCompletion(Task):
    summary = ("Supply the correctly inflected next word despite intervening nouns, "
               "coordination, and nested relative clauses, over singular versus "
               "plural heads and compound subjects from varied verb and noun banks.")
    config_cls = AgreementCompletionV2Config

    def generate_entry(self):
        cfg = self.config
        n_mods = cfg.n_mods

        if random.random() < cfg.compound_prob:
            n1 = random.choice(SING)
            n2 = random.choice(PLUR if random.random() < 0.6 else SING)
            d1 = random.choice(DETS_SING)
            d2 = random.choice(("the", "several", "a"))
            head_phrase = "%s %s and %s %s" % (d1, n1, d2, n2)
            agreement = "plural"
        else:
            if random.random() < 0.5:
                head_phrase = "%s %s" % (
                    random.choice(DETS_SING), random.choice(SING))
                agreement = "singular"
            else:
                head_phrase = "%s %s" % (
                    random.choice(DETS_PLUR), random.choice(PLUR))
                agreement = "plural"

        mods = [_modifier_block() for _ in range(n_mods)]
        subject = " ".join([head_phrase] + mods)

        base, sg3 = random.choice(VERBS)
        expected = sg3 if agreement == "singular" else base
        complement = random.choice(COMPLEMENTS)

        assert (expected == base) == (agreement == "plural"), (
            "verb form does not match the head's agreement number")

        metadata = edict(payload={
            "sentence_before": subject,
            "base_verb": base,
            "sentence_after": complement,
            "agreement": agreement,
        })
        metadata.answer = expected
        return Entry(metadata=metadata, answer=expected)

    def render_prompt(self, metadata):
        b = metadata.payload
        sentence = "%s  ____  %s." % (b["sentence_before"], b["sentence_after"])
        return (
            "Choose the verb that agrees with the grammatical subject of the clause, "
            "ignoring any nouns that merely modify that subject. Complete the sentence "
            "below by filling the blank with the correctly inflected present-tense form "
            "of the verb shown in brackets.\n"
            "Sentence: %s (verb: %s)\n"
            "\nWrite the answer as a single word: the inflected verb that fills the blank."
            % (sentence, b["base_verb"])
        )

    def score_answer(self, answer, entry):
        got = str(answer).strip().lower()
        expected = str(entry.answer).strip().lower()
        return 1.0 if got == expected else 0.0
