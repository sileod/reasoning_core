import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload

TASK_META = {'parent_source_id': None,
 'idea': 'minimal_edit (draw 1 of 2)',
 'hypothesis': 'ASTRA0-04',
 'changes': 'new task in reasoning_core/tasks/generated/wave10/minimal_edit',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3529541070,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


SUBJECTS = ["the chef", "the baker", "the pilot", "the lawyer", "the artist",
            "the engineer", "the sailor", "the guard", "the keeper", "the miner",
            "the tailor", "the drummer", "the weaver", "the herder", "the scout"]
VERBS = ["bake", "chop", "repair", "polish", "inspect", "capture", "deliver",
         "plant", "stir", "coil", "hoist", "weld"]
OBJECTS = ["cake", "fence", "engine", "mirror", "parcel", "lantern", "rope",
           "ladder", "barrel", "stove", "wagon", "harness"]
ADJS = ["fresh", "rusty", "heavy", "quiet", "gloomy", "crisp", "sturdy", "dim",
        "stale", "mossy", "shiny", "bulky"]
PLACES = ["at the inn", "in the shed", "near the dock", "by the river",
          "behind the mill", "under the awning", "at the market", "in the barn"]
REL_CLAUSES = ["who lives here", "who tends the fire", "who guards the gate",
               "who hums quietly", "who mends the nets", "who works overtime",
               "who keeps the ledger", "who reads aloud"]


@dataclass
class MinimalEditConfig(Config):
    n_mods: int = 0
    pool_size: int = 15

    def apply_difficulty(self, level):
        self.pool_size = 15 + level * 6
        self.n_mods = (level + 1) // 2


def _pick(rng, pool, used=None):
    while True:
        x = rng.choice(pool)
        if used is None or x not in used:
            return x


def _build_parts(cfg, rng=random):
    subj = _pick(rng, SUBJECTS)
    verb = _pick(rng, VERBS)
    obj = _pick(rng, OBJECTS)
    n_mods = cfg.n_mods
    mods = []
    adj = None
    if n_mods >= 1:
        mods.append(_pick(rng, PLACES))
    if n_mods >= 2:
        adj = _pick(rng, ADJS)
    if n_mods >= 3:
        mods.append(_pick(rng, REL_CLAUSES))
    return subj, verb, obj, mods, adj


def _assemble_positive(subj, verb, obj, mods, adj=None, rng=random):
    # returns full positive sentence string
    mods = list(mods)
    rel = None
    if any("who " in m for m in mods):
        rel = [m for m in mods if "who " in m][0]
        mods = [m for m in mods if "who " not in m]
    if rel:
        subj_ph = "{}, {},".format(subj, rel)
    else:
        subj_ph = subj
    parts = [subj_ph, verb + "s"]
    head = "the {} {}".format(adj, obj) if adj else "the {}".format(obj)
    parts.append(head)
    for m in mods:
        parts.append(m)
    return "{}.".format(" ".join(parts))


def _assemble_negative(subj, verb, obj, mods, adj=None, rng=random):
    mods = list(mods)
    rel = None
    if any("who " in m for m in mods):
        rel = [m for m in mods if "who " in m][0]
        mods = [m for m in mods if "who " not in m]
    if rel:
        subj_ph = "{}, {},".format(subj, rel)
    else:
        subj_ph = subj
    parts = [subj_ph, "does not", verb]
    head = "the {} {}".format(adj, obj) if adj else "the {}".format(obj)
    parts.append(head)
    for m in mods:
        parts.append(m)
    return "{}.".format(" ".join(parts))


class MinimalEdit(Task):
    summary = "Negate present-tense action sentences of varied structure (subject with optional relative clause, plain or adjectival object, place modifier), flipping between positive and negative polarity by inserting or removing a 'does not' on the main verb while preserving every other word."
    config_cls = MinimalEditConfig
    task_version = 2

    def generate_entry(self):
        cfg = self.config
        subj, verb, obj, mods, adj = _build_parts(cfg)
        positive = _assemble_positive(subj, verb, obj, list(mods), adj)
        negative = _assemble_negative(subj, verb, obj, list(mods), adj)
        is_negated = random.randint(0, 1) == 1
        source = negative if is_negated else positive
        gold = positive if is_negated else negative
        # constructive verifier: flipping the gold polarity must reproduce the source,
        # and flipping the source must reproduce the gold.
        if is_negated:
            recomputed = _assemble_positive(subj, verb, obj, list(mods), adj)
            assert recomputed == gold, (recomputed, gold)
            back = _assemble_negative(subj, verb, obj, list(mods), adj)
            assert back == source, (back, source)
        else:
            recomputed = _assemble_negative(subj, verb, obj, list(mods), adj)
            assert recomputed == gold, (recomputed, gold)
            back = _assemble_positive(subj, verb, obj, list(mods), adj)
            assert back == source, (back, source)
        assert gold != source
        metadata = edict({
            "sentence": source,
            "polarity": "positive" if not is_negated else "negative",
            "answer": gold,
        })
        metadata.payload = {"sentence": source}
        return Entry(metadata=metadata, answer=gold)

    def render_prompt(self, metadata):
        sentence = metadata.sentence
        return (
            f"Rewrite the following sentence so that it has the opposite negation: "
            f"if it is currently positive, negate it by inserting 'does not' before the "
            f"main verb (returning that verb to its base form); if it already contains "
            f"'does not', remove it and restore the verb's -s ending. Change no other words, "
            f"capitalisation, or punctuation, and keep the relative clause and any modifiers "
            f"exactly where they are.\n\nSentence: {sentence}\n\n"
            f"Give only the rewritten sentence as the answer."
        )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        gold = normalize(entry.answer)
        if not gold:
            return 0.0
        a = normalize(answer)
        if not a:
            return 0.0
        return 1.0 if a == gold else 0.0


def normalize(s):
    if not isinstance(s, str):
        return ""
    t = s.strip().rstrip(".").strip()
    t = t.replace("doesn't", "does not")
    t = t.replace("doesnt", "does not")
    t = t.replace("isn't", "is not")
    t = t.replace(" isn't ", " is not ")
    return " ".join(t.split()).lower()
