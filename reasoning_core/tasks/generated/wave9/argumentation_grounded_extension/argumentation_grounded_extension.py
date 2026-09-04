from dataclasses import dataclass

import random

from reasoning_core.template import Config, Entry, Task, edict

TASK_META = {'parent_source_id': None,
 'idea': 'argumentation_grounded_extension (draw 1 of 1)',
 'hypothesis': 'HV-053',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/argumentation_grounded_extension',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 4070707122,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class ArgumentationGroundedExtensionConfig(Config):
    n_args: int = 5
    p_attack: float = 0.20

    def apply_difficulty(self, level):
        n = self.n_args + 2 * level
        self.n_args = int(n)
        self.p_attack = 0.20


def _grounded(names, att):
    """Least fixed point of the acceptability characteristic function (Dung 1995).

    att is a set of (attacker, victim) pairs. An argument is acceptable w.r.t. a
    set S when every argument attacking it is itself attacked by a member of S.
    The grounded extension is the least fixed point reached by iteration.
    """
    name_set = set(names)
    fixed = set()
    while True:
        nxt = set()
        for a in names:
            attackers = [b for b in names if (b, a) in att]
            if all(any((s, b) in att for s in fixed) for b in attackers):
                nxt.add(a)
        if nxt == fixed:
            break
        fixed = nxt
    return fixed


def _grounded_labeling(names, att):
    """Independent grounded-labeling check: IN/OUT/UNDEC labels to a fixed point."""
    att_set = set(att)
    in_set = set()
    out_set = set()
    undec = set(names)
    while True:
        progressed = False
        for a in list(undec):
            attackers = [b for b in names if (b, a) in att_set]
            if all(b in out_set for b in attackers):
                undec.discard(a)
                in_set.add(a)
                progressed = True
        for a in list(undec):
            attackers = [b for b in names if (b, a) in att_set]
            if any(b in in_set for b in attackers):
                undec.discard(a)
                out_set.add(a)
                progressed = True
        if not progressed:
            break
    return in_set


def _parse_answer(answer):
    if answer is None:
        return frozenset()
    tokens = str(answer).strip().lower().split()
    if not tokens or tokens == ["none"]:
        return frozenset()
    return frozenset(tokens)


def _forced_graph(names):
    """A guaranteed non-trivial graph: x attacks y, y attacks z, no one attacks x.

    x has no attackers so it is accepted, y is rejected, z is defended (its only
    attacker y is rejected) so z is accepted. The grounded extension is therefore
    always non-empty (contains x, z and any isolated rest) and never the full set
    (y is always rejected).
    """
    x, y, z = names[0], names[1], names[2]
    order = [(x, y), (y, z)]
    return set(order), sorted(order)


class ArgumentationGroundedExtension(Task):
    summary = ("Compute the grounded extension of abstract argumentation graphs with "
               "varied argument counts, attack densities, and defense chains by "
               "iterating attack and defense relations to a fixed point.")
    config_cls = ArgumentationGroundedExtensionConfig
    task_version = 2

    def generate_entry(self):
        n = int(self.config.n_args)
        p = float(self.config.p_attack)
        names = [chr(ord("a") + i) for i in range(n)]
        for _ in range(1000):
            att = set()
            order = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if random.random() < p:
                        order.append((names[i], names[j]))
                        att.add((names[i], names[j]))
            order = sorted(order)
            answer_set = _grounded(names, att)
            ck = _grounded_labeling(names, att)
            if answer_set != ck:
                raise RuntimeError("grounded implementations disagree")
            if not answer_set or len(answer_set) == n:
                continue
            answer = " ".join(sorted(answer_set))
            metadata = edict({
                "n_args": n,
                "p_attack": p,
                "payload": {
                    "arguments": " ".join(names),
                    "attacks": "\n".join(f"{x} attacks {y}" for x, y in order),
                },
            })
            return Entry(metadata=metadata, answer=answer)
        att, order = _forced_graph(names)
        answer_set = _grounded(names, att)
        ck = _grounded_labeling(names, att)
        if answer_set != ck:
            raise RuntimeError("grounded implementations disagree")
        if not answer_set or len(answer_set) == n:
            raise RuntimeError("forced graph not non-trivial")
        answer = " ".join(sorted(answer_set))
        metadata = edict({
            "n_args": n,
            "p_attack": p,
            "payload": {
                "arguments": " ".join(names),
                "attacks": "\n".join(f"{x} attacks {y}" for x, y in order),
            },
        })
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        payload = metadata.payload
        return (
            f"Consider the abstract argumentation framework with arguments: "
            f"{payload['arguments']}.\n"
            f"Attacks (X attacks Y means the argument X attacks Y):\n"
            f"{payload['attacks']}\n\n"
            f"Using grounded semantics, compute the grounded extension by repeatedly "
            f"marking an argument accepted whenever every argument attacking it is "
            f"already rejected, and rejected whenever some accepted argument attacks "
            f"it, until nothing changes. Name the arguments in the grounded extension, "
            f"written as a single space-separated sequence in alphabetical order "
            f"(write none if the grounded extension is empty)."
        )

    def score_answer(self, answer, entry):
        gold = frozenset(str(entry.answer).split())
        return 1.0 if _parse_answer(answer) == gold else 0.0
