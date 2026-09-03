import random
import re
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload

TASK_META = {'parent_source_id': None,
 'idea': 'c3_linearization (draw 2 of 2)',
 'hypothesis': 'W1-054',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/c3_linearization',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1493643473,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

_CLASS_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _class_name(i):
    return _CLASS_LETTERS[i % len(_CLASS_LETTERS)]


def c3_linearize(name, bases, mro):
    """C3 merge over the parents' linearizations; raises ValueError if inconsistent."""
    seqs = [list(mro[b]) for b in bases] + [list(bases)]
    result = [name]
    while any(seqs):
        seqs = [s for s in seqs if s]
        if not seqs:
            break
        chosen = None
        for s in seqs:
            head = s[0]
            if not any(head in other[1:] for other in seqs):
                chosen = s
                break
        if chosen is None:
            raise ValueError("inconsistent hierarchy")
        head = chosen[0]
        result.append(head)
        for s in seqs:
            if s and s[0] == head:
                s.pop(0)
    if len(result) != len(set(result)):
        raise ValueError("duplicate class in linearization")
    return result


@dataclass
class C3LinearizationConfig(Config):
    def apply_difficulty(self, level):
        pass


class C3Linearization(Task):
    summary = ("Given a multiple-inheritance DAG, output the C3 method-resolution order of a "
               "queried class; level scales class count and base-arity.")
    config_cls = C3LinearizationConfig
    task_version = 2

    def generate_entry(self):
        level = self.config.level
        total = 7 + 2 * level
        max_bases = 2 + level

        names = [_class_name(i) for i in range(total)]
        mro = {}
        parents = {}
        root = names[0]
        mro[root] = [root]
        parents[root] = []

        for i in range(1, total):
            name = names[i]
            bases = [names[i - 1]]
            n_extra = random.randint(0, max_bases - 1) if random.random() < 0.65 else 0
            tried = 0
            while len(bases) - 1 < n_extra and tried < 60:
                tried += 1
                cand = random.choice(names[:i])
                if cand in bases:
                    continue
                probe = bases + [cand]
                try:
                    c3_linearize(name, probe, mro)
                except ValueError:
                    continue
                bases = probe
            L = c3_linearize(name, bases, mro)
            mro[name] = L
            parents[name] = list(bases)

        eligible = [c for c in names[1:] if len(mro[c]) >= 3]
        if not eligible:
            raise RuntimeError("no queriable class produced")
        query = random.choice(eligible)
        answer = ", ".join(mro[query])

        lines = []
        for name in names:
            if parents[name]:
                lines.append(f"class {name}({', '.join(parents[name])})")
        class_block = "\n".join(lines)
        query_block = f"class {query}"

        metadata = edict({
            "classes": class_block,
            "query": query_block,
            "n_classes": total,
            "mro": mro[query],
            "parents": {k: list(v) for k, v in parents.items()},
        })
        metadata.payload = {
            "Classes": class_block,
            "Query": query_block,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (
            f"{render_payload(metadata.payload)}\n\n"
            "Each line declares a class and its direct base classes, in C3 (Dylan/C3) "
            "linearization semantics. Compute the C3 method-resolution order (MRO) of the queried "
            "class: it starts with that class itself, then merges the parent linearizations, "
            "most-derived class first, ending at the least-derived ancestor.\n\n"
            "Output only the MRO as a single line: the class names in linearization order, "
            "separated by commas. For example, if the MRO is X then Y then Z, the answer is "
            "'X, Y, Z'.\n\n"
            "The answer is the comma-separated MRO of the queried class."
        )

    def score_answer(self, answer, entry):
        norm = lambda s: re.sub(r"\s+", "", str(s))
        return 1.0 if norm(answer) == norm(entry.answer) else 0.0
