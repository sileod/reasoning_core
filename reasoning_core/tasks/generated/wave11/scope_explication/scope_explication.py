"""Explicit-scope disambiguation: rewrite negation/quantifier/modifier ambiguities.

A sentence carries an ambiguous negation, quantifier, or modifier and a bracketed
note pinning the intended reading. The solver must emit the corresponding short,
unambiguous restatement. Clauses are drawn from three families (universal
negation, existential negation, and adjective-modifier scope), each answerable
under two distinct readings, and chained so difficulty grows with clause count.
"""

import random

from reasoning_core.template import Config, Entry, Task, edict, render_payload

SUBJ = ["chefs", "judges", "hikers", "sailors", "villagers", "students", "artists", "doctors"]
OBJ = ["coin", "dish", "lighthouse", "book", "flower", "painting", "gem", "medal"]
VERBS = [
    ("taste", "tasted", "tasted"),
    ("see", "saw", "seen"),
    ("find", "found", "found"),
    ("read", "read", "read"),
    ("buy", "bought", "bought"),
    ("catch", "caught", "caught"),
    ("open", "opened", "opened"),
    ("clean", "cleaned", "cleaned"),
]
NOISE1 = ["painting", "picture", "sketch", "portrait"]
N2 = ["cat", "dog", "horse", "bridge", "house"]
ADJ = ["small", "large", "red", "old", "strange", "clean"]

SEP = " and "

# family key -> (surface, reading_hint, answer) builders given random picks
_FAMILIES = ["A", "B", "C"]


def _clause_a(subj, vb, obj, neg_wide):
    base, past, _ = vb
    surface = "All the %s didn't %s a %s." % (subj, base, obj)
    if neg_wide:
        hint = ("The negation covers the whole sentence, so it is not the case "
                "that all the %s %s a %s." % (subj, past, obj))
        ans = "Some %s didn't %s a %s" % (subj, base, obj)
    else:
        hint = ("The negation covers only the action, so all the %s failed to %s "
                "any %s." % (subj, base, obj))
        ans = "No %s %s a %s" % (subj, past, obj)
    assert "not the case that all the" in hint or "failed to" in hint
    return surface, hint, ans


def _clause_b(subj, vb, obj, neg_exists):
    base, past, pp = vb
    surface = "The %s didn't %s a %s." % (subj, base, obj)
    if neg_exists:
        hint = ("The negation covers the whole claim, so it is false that any %s "
                "was %s by the %s." % (obj, pp, subj))
        ans = "No %s was %s by the %s" % (obj, pp, subj)
    else:
        hint = "There is a %s that the %s did not %s." % (obj, subj, base)
        ans = "At least one %s was not %s by the %s" % (obj, pp, subj)
    assert "any %s was" % obj in hint or "There is a" in hint
    return surface, hint, ans


def _clause_c(noise1, n2, adj, adj_first):
    surface = "Nina drew a %s of a %s that was %s." % (noise1, n2, adj)
    if adj_first:
        hint = "The word %s describes the %s." % (adj, noise1)
        ans = "Nina drew a %s %s of a %s" % (adj, noise1, n2)
    else:
        hint = "The word %s describes the %s." % (adj, n2)
        ans = "Nina drew a %s of a %s %s" % (noise1, adj, n2)
    assert noise1 != n2
    return surface, hint, ans


def _make_clause():
    family = random.choice(_FAMILIES)
    if family == "A":
        return _clause_a(
            random.choice(SUBJ), random.choice(VERBS), random.choice(OBJ),
            neg_wide=random.random() < 0.5)
    if family == "B":
        return _clause_b(
            random.choice(SUBJ), random.choice(VERBS), random.choice(OBJ),
            neg_exists=random.random() < 0.5)
    return _clause_c(
        random.choice(NOISE1), random.choice(N2), random.choice(ADJ),
        adj_first=random.random() < 0.5)


def _norm(s):
    s = str(s).strip().lower()
    for token in (" and ", ",", ";", ".", "!", "?", "(", ")"):
        s = s.replace(token, " ")
    return " ".join(s.split())


class ScopeExplicationConfig(Config):
    n_clauses: int = 1

    def apply_difficulty(self, level):
        self.n_clauses = 1 + level // 2


class ScopeExplication(Task):
    summary = ("Rewrite explicitly disambiguated negation, quantifiers, and modifiers "
               "into a short unambiguous statement across universal/existential negation "
               "and adjective-modifier scope, each under two readings, chained by clause count.")
    config_cls = ScopeExplicationConfig
    task_version = 2

    def generate_entry(self):
        surfaces, hints, answers = [], [], []
        for _ in range(self.config.n_clauses):
            s, h, a = _make_clause()
            surfaces.append(s)
            hints.append(h)
            answers.append(a)

        payload = {
            "sentence": " ".join(surfaces) + " Here " + " ".join(
                ["(%s)" % h for h in hints]),
        }
        answer = SEP.join(answers)
        metadata = edict({"payload": payload, "answer": answer, "surface": " ".join(surfaces)})
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        header = (
            "A sentence carries an ambiguous negation, quantifier, or modifier. "
            "A parenthetical note states the intended reading. Rewrite it as the "
            "short unambiguous statement that captures exactly that reading.\n\n"
            "Example: Sentence: Every chef didn't taste a dish. (The negation covers "
            "the whole sentence, so it is not the case that every chef tasted a dish.)\n"
            "Answer: Some chef didn't taste a dish.\n\n"
        )
        return header + render_payload(metadata.payload) + "\n\nThe answer is the short unambiguous statement."

    def score_answer(self, answer, entry):
        return 1.0 if _norm(answer) == _norm(entry.answer) else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'scope_explication (draw 1 of 2)',
 'hypothesis': 'ASTRA2-scope_explication',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave11/scope_explication',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 974528455,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
