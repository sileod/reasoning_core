import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'fd_attribute_closure (draw 1 of 2)',
 'hypothesis': 'W1-027',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/fd_attribute_closure',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2514553751,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class AttributeClosureConfig(Config):
    n_atts: int = 7
    n_fds: int = 4
    max_lhs: int = 3
    max_rhs: int = 3

    def apply_difficulty(self, level):
        self.n_atts = sround(self.n_atts + level)
        self.n_fds = sround(self.n_fds + level)
        self.max_lhs = sround(self.max_lhs + (level > 2))
        self.max_rhs = sround(self.max_rhs + (level > 2))


def _closure(fds, start):
    closure = set(start)
    changed = True
    while changed:
        changed = False
        for lhs, rhs in fds:
            if lhs.issubset(closure):
                new = rhs - closure
                if new:
                    closure |= new
                    changed = True
    return closure


def _subset(rng_attrs, k):
    return frozenset(random.sample(rng_attrs, k))


def _nonempty_subset(rng_attrs, max_k):
    k = random.randint(1, max_k)
    return _subset(rng_attrs, k)


class AttributeClosure(Task):
    task_name = "fd_attribute_closure"
    summary = "Given functional dependencies and attributes, compute the attribute closure of a starting set via Armstrong iteration and output it as a sorted letter string."

    config_cls = AttributeClosureConfig

    def generate_entry(self):
        cfg = self.config
        n = cfg.n_atts
        attrs = list(range(n))
        names = [chr(ord("A") + i) for i in range(n)]

        for _ in range(300):
            n_fds = max(1, cfg.n_fds)
            fds = []
            for _ in range(n_fds):
                lhs = _nonempty_subset(attrs, cfg.max_lhs)
                rhs = _nonempty_subset(attrs, cfg.max_rhs)
                fds.append((lhs, rhs))

            start_size = random.randint(1, 2)
            start = frozenset(random.sample(attrs, start_size))

            clos = _closure(fds, start)

            closure = keep = None
            # reject trivial / surface-readable answers
            surface = set()
            for lhs, rhs in fds:
                surface |= lhs | rhs
            if clos == start:
                continue
            if clos == surface:
                continue
            closure = set(clos)
            keep = (fds, start, closure)
            break
        else:
            raise RuntimeError("could not build a non-trivial closure instance")

        fds, start, closure = keep

        fd_lines = []
        for lhs, rhs in fds:
            lhs_s = "".join(sorted(names[i] for i in lhs))
            rhs_s = "".join(sorted(names[i] for i in rhs))
            fd_lines.append(f"{lhs_s} -> {rhs_s}")

        start_s = "{" + ", ".join(sorted(names[i] for i in start)) + "}"
        universe_s = "{ " + ", ".join(names) + " }"
        answer = "".join(sorted(names[i] for i in closure))

        metadata = edict({
            "payload": {
                "universe": universe_s,
                "dependencies": "\n".join(fd_lines),
                "start": start_s,
            },
            "n_atts": n,
        })
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        payload = render_payload(metadata.payload)
        return (
            f"{payload}\n\n"
            "Using the standard attribute-closure algorithm (repeatedly apply every "
            "functional dependency whose left side is already implied), compute the "
            "attribute closure of the starting set under these functional dependencies.\n\n"
            "The answer is the closure written as a single string: its attribute letters "
            "concatenated, with no separators, sorted in alphabetical order. "
            "For example, a closure containing attributes A, C and B would be written ACB "
            "reordered as ABC."
        )

    def score_answer(self, answer, entry):
        try:
            norm = "".join(ch for ch in answer.upper() if ch.isalpha())
        except (TypeError, AttributeError):
            return 0.0
        return 1.0 if norm == entry.answer else 0.0
