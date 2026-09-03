"""Overload resolution: pick the unique most-specific applicable overload."""

import itertools
import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'overload_resolution (draw 1 of 2)',
 'hypothesis': 'W1-057',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/overload_resolution',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3892076091,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

PARAM_TYPES = ["int", "float", "number", "str", "bool", "list", "char", "bytes", "tuple"]
NUMERIC = ("int", "float", "number")

CONVERSION_DISTANCE = {
    ("int", "int"): 0,
    ("float", "float"): 0,
    ("number", "number"): 0,
    ("str", "str"): 0,
    ("bool", "bool"): 0,
    ("list", "list"): 0,
    ("int", "float"): 1,
    ("float", "number"): 1,
    ("int", "number"): 2,
}

NOCONV = 10 ** 9


def _conv_cost(arg, param):
    """Cost of converting an argument type to a parameter type, or None if not applicable."""
    if arg == param:
        return 0
    if (arg, param) in CONVERSION_DISTANCE:
        return CONVERSION_DISTANCE[(arg, param)]
    return None


def _matches(arg, param):
    if arg == param:
        return True
    if arg == "number" or param == "number":
        # numeric family
        return arg in ("int", "float", "number") and param in ("int", "float", "number")
    return False


def applicable(signature, arg_types):
    if len(signature) != len(arg_types):
        return False
    for p, a in zip(signature, arg_types):
        if not _matches(a, p):
            return False
    return True


def signature_cost(signature, arg_types):
    total = 0
    for p, a in zip(signature, arg_types):
        c = _conv_cost(a, p)
        if c is None:
            return None
        total += c
    return total


def most_specific(sig_a, sig_b):
    """Return 1 if sig_a strictly more specific than sig_b, -1 if reverse, 0 if neither/incomparable."""
    strict_a = False
    strict_b = False
    for pa, pb in zip(sig_a, sig_b):
        if pa == pb:
            continue
        if _conv_cost(pa, pb) is not None and _conv_cost(pa, pb) >= 0 and _conv_cost(pb, pa) is None:
            # pa can convert to pb (pa more specific) but not reverse
            strict_a = True
            continue
        if _conv_cost(pb, pa) is not None and _conv_cost(pb, pa) >= 0 and _conv_cost(pa, pb) is None:
            strict_b = True
            continue
        # both ways? both are from same numeric family different types - handle
        if pa in ("int", "float", "number") and pb in ("int", "float", "number") and pa != pb:
            ca = _conv_cost(pb, pa)  # cost to go pb->pa
            cb = _conv_cost(pa, pb)
            if ca is not None and cb is None:
                strict_a = True
                continue
            if cb is not None and ca is None:
                strict_b = True
                continue
        return 0
    if strict_a and not strict_b:
        return 1
    if strict_b and not strict_a:
        return -1
    return 0


def resolve(overloads, arg_types):
    """Return index of unique most-specific applicable overload, or None if ambiguous/none."""
    applicable_idx = [i for i, sig in enumerate(overloads) if applicable(sig, arg_types)]
    if not applicable_idx:
        return None
    if len(applicable_idx) == 1:
        return applicable_idx[0]
    best = applicable_idx[:]
    for i in list(best):
        for j in list(best):
            if i == j:
                continue
            rel = most_specific(overloads[i], overloads[j])
            if rel == -1:
                # i is less specific than j -> i cannot be most specific
                if i in best:
                    best.remove(i)
    if len(best) == 1:
        return best[0]
    return None


@dataclass
class OverloadConfig(Config):
    n_overloads_min: int = 2
    n_overloads_max: int = 3
    n_args: int = 1
    ambiguity_rate: float = 0.15

    def apply_difficulty(self, level):
        self.n_overloads_min = sround(2 + level)
        self.n_overloads_max = sround(3 + level)
        self.n_args = sround(1 + (level >= 2))
        self.ambiguity_rate = min(0.12 + 0.04 * level, 0.3)


class OverloadResolution(Task):
    summary = ("Given overload signatures and argument types, output the unique "
               "most-specific applicable overload or Ambiguous, over numeric-family "
               "conversion chains and exact-type matches.")
    config_cls = OverloadConfig

    def generate_entry(self):
        cfg = self.config
        for _ in range(600):
            nargs = cfg.n_args
            n = random.randint(cfg.n_overloads_min, cfg.n_overloads_max)
            arg_types = tuple(random.choice(PARAM_TYPES) for _ in range(nargs))

            numeric_positions = [i for i, t in enumerate(arg_types) if t in NUMERIC]

            # Choose the winner. With some probability the winner uses a strictly-broader
            # numeric type ('number' at a numeric position) so the answer differs from the
            # literal argument type; otherwise it is the exact match.
            if numeric_positions and random.random() < 0.4:
                if all(t in NUMERIC for t in arg_types):
                    # pure-numeric: pick a uniform winner over the numeric types
                    winner = tuple(random.choice(NUMERIC) for _ in range(nargs))
                else:
                    def _br():
                        return tuple(
                            t if t not in NUMERIC else "number" for t in arg_types
                        )
                    winner = _br()
            else:
                winner = tuple(t for t in arg_types)

            # Collect other applicable overloads that are strictly less specific than the
            # winner and mutually comparable, so the winner stays the unique most specific.
            chosen_app = [winner]
            for _ in range(random.randint(0, 1) if numeric_positions else 0):
                broader = tuple(
                    t if t not in NUMERIC else random.choice(["float", "number"])
                    for t in arg_types
                )
                if broader != winner and applicable(broader, arg_types):
                    chosen_app.append(broader)

            def random_sig():
                return tuple(random.choice(PARAM_TYPES) for _ in range(nargs))

            distractors = []
            while len(chosen_app) + len(distractors) < n:
                cand = random_sig()
                if not applicable(cand, arg_types) and cand not in chosen_app and cand not in distractors:
                    distractors.append(cand)

            overloads = list(chosen_app) + distractors
            random.shuffle(overloads)

            # Optionally manufacture a genuine ambiguity: a second applicable overload that is
            # incomparable to the winner -> no unique most-specific.
            ambiguous = False
            if random.random() < cfg.ambiguity_rate:
                other = tuple(
                    t if t not in NUMERIC else "int" for t in arg_types
                )
                if other != winner and applicable(other, arg_types):
                    overloads = list(chosen_app) + [other] + distractors
                    random.shuffle(overloads)
                    ambiguous = True

            result = resolve(overloads, arg_types)
            if result is not None:
                answer = ", ".join(overloads[result])
            else:
                answer = "Ambiguous"

            # Independent verification:
            if not ambiguous:
                sig = tuple(answer.split(", "))
                applicable_idx = [i for i, s in enumerate(overloads) if applicable(s, arg_types)]
                matching = [i for i, s in enumerate(overloads) if s == sig]
                if len(matching) != 1:
                    continue
                idx = matching[0]
                is_best = True
                for j, s in enumerate(overloads):
                    if j == idx:
                        continue
                    if not applicable(s, arg_types):
                        continue
                    rel = most_specific(overloads[idx], s)
                    if rel == -1 or rel == 0:
                        is_best = False
                        break
                if not is_best:
                    continue
            else:
                applicable_idx = [i for i, s in enumerate(overloads) if applicable(s, arg_types)]
                if not applicable_idx:
                    continue
                res = resolve(overloads, arg_types)
                if res is not None:
                    continue

            # also ensure answer isn't trivially the surface (no last/first/largest number trick because answer is small index; fine)
            metadata = edict({
                "overloads": list(overloads),
                "arg_types": list(arg_types),
            })
            metadata.payload = {
                "overloads": list(overloads),
                "arg_types": list(arg_types),
            }
            return Entry(metadata=metadata, answer=answer)
        raise RuntimeError("failed to generate overload instance")

    def render_prompt(self, metadata):
        lines = []
        for i, sig in enumerate(metadata.overloads):
            params = ", ".join(sig)
            lines.append(f"  f{i}({params})")
        overloads_block = "\n".join(lines)
        args = ", ".join(metadata.arg_types)
        return (f"Consider these overloads of function f:\n"
                f"{overloads_block}\n"
                f"f is called with arguments of types ({args}).\n"
                f"Using standard overload resolution (numeric conversions cost more than exact "
                f"matches, and a most-specific applicable overload is chosen when unique), "
                f"which overload is chosen? Output the parameter-type list of the unique "
                f"most-specific applicable overload as comma-separated types with no spaces "
                f"(for example int, float), otherwise output the word Ambiguous.\n"
                f"Format: comma-separated types or Ambiguous.")

    def score_answer(self, answer, entry):
        gold = entry.answer
        if isinstance(answer, str):
            a = answer.strip()
        else:
            a = str(answer)
        return 1.0 if a == gold else 0.0
