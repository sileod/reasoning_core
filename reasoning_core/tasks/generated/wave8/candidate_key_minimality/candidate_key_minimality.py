import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'candidate_key_minimality (draw 1 of 2)',
 'hypothesis': 'W1-028',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/candidate_key_minimality',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3142999616,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

LABELS = ["non-superkey", "nonminimal superkey", "candidate key"]


def compute_closure(start, fds, all_attrs):
    cur = set(start)
    changed = True
    while changed:
        changed = False
        for lhs, rhs in fds:
            if lhs <= cur and not rhs <= cur:
                cur |= rhs
                changed = True
    return cur


def _frozen(seq_of_fds):
    return [(frozenset(l), frozenset(r)) for l, r in seq_of_fds]


@dataclass
class CandidateKeyMinimalityConfig(Config):
    n_attrs: int = 6

    def apply_difficulty(self, level):
        self.n_attrs = sround(6 + level)


class CandidateKeyMinimality(Task):
    summary = ("Classify an attribute set under functional dependencies as "
               "non-superkey, nonminimal superkey, or candidate key.")
    config_cls = CandidateKeyMinimalityConfig

    def generate_entry(self):
        n = int(self.config.n_attrs)
        attrs = ["A%d" % i for i in range(n)]
        r = random.randint(2, (n - 1) // 2)
        d = n - r
        key_attrs = attrs[:r]
        dep_attrs = attrs[r:]
        fds = []
        for j in range(d):
            src = key_attrs[j % r]
            fds.append((sorted([src]), sorted([dep_attrs[j]])))

        key_set = set(key_attrs)
        dep_set = set(dep_attrs)
        all_set = set(attrs)

        label = random.choice(LABELS)

        if label == "candidate key":
            x = sorted(key_attrs)
        elif label == "nonminimal superkey":
            spare = dep_attrs[r]
            x = sorted(key_attrs + [spare])
        else:
            m = random.randint(1, d)
            chosen = sorted(random.sample(dep_attrs, m))
            x = chosen

        frozen_fds = _frozen(fds)
        closure = compute_closure(x, frozen_fds, all_set)

        if label == "candidate key":
            assert closure == all_set
        elif label == "nonminimal superkey":
            assert closure == all_set
            assert compute_closure(key_attrs, frozen_fds, all_set) == all_set
        else:
            assert closure < all_set

        if closure == all_set:
            is_minimal = all(
                compute_closure(sorted([a for a in x if a != t]), frozen_fds, all_set) != all_set
                for t in x
            )
            true_label = "candidate key" if is_minimal else "nonminimal superkey"
        else:
            true_label = "non-superkey"
        assert true_label == label

        payload = {
            "attributes": attrs,
            "functional_dependencies": ["%s -> %s" % (l[0], rhs[0]) for l, rhs in fds],
            "set": x,
        }
        metadata = edict({})
        metadata.payload = payload
        return Entry(metadata=metadata, answer=true_label)

    def render_prompt(self, metadata):
        p = render_payload(metadata.payload)
        return (p
                + "\n\nA superkey is a set of attributes whose closure contains all "
                  "attributes; a candidate key is a minimal superkey. Classify the given "
                  "attribute set with respect to the functional dependencies. The answer is "
                  "exactly one of: 'candidate key', 'nonminimal superkey', 'non-superkey'.")

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        return 1.0 if answer.strip() == entry.answer else 0.0
