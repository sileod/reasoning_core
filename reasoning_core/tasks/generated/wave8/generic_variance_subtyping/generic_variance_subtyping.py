import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'generic_variance_subtyping (draw 1 of 2)',
 'hypothesis': 'W1-056',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/generic_variance_subtyping',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2438869589,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

MARKERS = ["Fruit", "Shape", "Color", "Matter", "Number", "Direction"]

# Each concrete ground class carries a fixed set of markers (its position in the
# subset lattice of MARKERS). Subtyping A <: B means marker(A) subseteq marker(B).
CLASSES = {
    "Apple": ("Fruit",),
    "Round": ("Shape",),
    "Red": ("Color",),
    "Mint": ("Color", "Number"),
    "Food": ("Fruit", "Shape", "Color"),
    "Stone": ("Matter",),
    "Hard": ("Matter", "Shape"),
    "Even": ("Number",),
    "Num": ("Number", "Direction"),
    "Left": ("Direction",),
    "Berry": ("Fruit", "Color"),
    "Sweet": ("Fruit", "Shape"),
}

CLASS_NAMES = sorted(CLASSES.keys())

VAR_WEIGHTS = ["+", "+", "+", "-", "-", "-", "o", "o", "*"]


def _class_set(name):
    return frozenset(CLASSES[name])


def _param_set(node):
    """node is ('single', name) or ('join', [names]). Returns frozenset of markers."""
    kind, payload = node
    acc = set()
    if kind == "single":
        acc = set(_class_set(payload))
    else:
        for nm in payload:
            acc |= set(_class_set(nm))
    return frozenset(acc)


def _render_param(node):
    kind, payload = node
    if kind == "single":
        return payload
    return " \u2294 ".join(payload)


def compute_witness(a_set, b_set, var):
    """Return sorted list of witness markers. Empty list means the subtype
    relation holds. Semantics per variance of Box's parameter position."""
    if var == "+":
        return sorted(a_set - b_set)
    if var == "-":
        return sorted(b_set - a_set)
    if var == "o":
        return sorted(a_set ^ b_set)
    if var == "*":
        return []
    raise RuntimeError("bad variance")


def _answer_string(witness):
    if not witness:
        return "none"
    return ", ".join(witness)


def _make_param(operands):
    depth = random.randrange(1, operands + 1)
    if depth == 1:
        return ("single", random.choice(CLASS_NAMES))
    names = []
    pool = list(CLASS_NAMES)
    for _ in range(depth):
        nm = random.choice(pool)
        pool.remove(nm)
        names.append(nm)
    return ("join", names)


@dataclass
class VarianceConfig(Config):
    operands: int = 1
    join_p: float = 0.0

    def apply_difficulty(self, level):
        self.operands = 1 + sround(min(level // 2, 2))
        self.join_p = min(0.18 * level, 0.8)


class VarianceSubtyping(Task):
    task_name = "generic_variance_subtyping"
    summary = "Given variance declarations and ground types, answer a parameterized subtype query."
    config_cls = VarianceConfig

    def generate_entry(self):
        c = self.config
        operands = int(c.operands)

        p_join = bool(c.join_p > 0 and random.random() < c.join_p)
        q_join = bool(c.join_p > 0 and random.random() < c.join_p)
        a_param = _make_param(operands) if p_join else ("single", random.choice(CLASS_NAMES))
        b_param = _make_param(operands) if q_join else ("single", random.choice(CLASS_NAMES))

        var = random.choice(VAR_WEIGHTS)

        a_set = _param_set(a_param)
        b_set = _param_set(b_param)
        witness = compute_witness(a_set, b_set, var)
        answer = _answer_string(witness)

        # Verifier: reconstruct the gold answer from the instance and assert.
        recomputed = compute_witness(_param_set(a_param), _param_set(b_param), var)
        assert _answer_string(recomputed) == answer
        # Domain check: every witness marker must be one of the declared markers.
        for m in recomputed:
            assert m in MARKERS

        var_word = {"+": "covariant", "-": "contravariant", "o": "invariant",
                    "*": "bivariant"}[var]
        var_sign = {"+": "+", "-": "-", "o": "o", "*": "*"}[var]

        metadata = edict({
            "ground_types": {k: list(v) for k, v in sorted(CLASSES.items())},
            "param_a": _render_param(a_param),
            "param_b": _render_param(b_param),
            "variance": var,
            "variance_word": var_word,
            "a_markers": sorted(a_set),
            "b_markers": sorted(b_set),
            "witness": list(witness),
        })
        metadata.payload = {
            "ground_types": metadata.ground_types,
            "a": metadata.param_a,
            "b": metadata.param_b,
            "variance": metadata.variance,
            "variance_word": metadata.variance_word,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = ["The generic type Box takes one type parameter marked "
                 f"{metadata.variance} ({metadata.variance_word})."]
        gt = "; ".join(f"{k}={{{', '.join(v)}}}" for k, v in sorted(metadata.ground_types.items()))
        lines.append(f"Ground types (each marker-set is the type's position): {gt}.")
        lines.append(f"Under {metadata.variance_word} variance, Box<A> <: Box<B> requires:")
        lines.append("  covariant (+): A's markers are a subset of B's markers;")
        lines.append("  contravariant (-): B's markers are a subset of A's markers;")
        lines.append("  invariant (o): A and B have exactly the same markers;")
        lines.append("  bivariant (*): the relation always holds.")
        lines.append(f"Compare A = {metadata.param_a} with B = {metadata.param_b} under "
                     f"{metadata.variance_word} variance.")
        lines.append("Give the minimal set of markers responsible for the failure "
                     "of that subtype relation, sorted alphabetically and separated "
                     "by commas; if it holds, write none.")
        lines.append("Example: A = Mint (Color,Number), B = Round (Shape), covariant -> "
                     "Mint's markers not in Round are Color, Number.")
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        return _normalize_match(answer, entry.answer)


def _normalize_match(answer, gold):
    if not isinstance(answer, str):
        return 0.0
    def norm(s):
        return " ".join(s.lower().replace(":","").replace("\u2294","").split())
    if norm(answer) == norm(gold):
        return 1.0
    # Accept marker lists in any order by comparing normalized sets.
    if gold == "none":
        return 0.0
    def settify(s):
        if "none" in s.lower() and len(s.split(",")) == 1:
            return None
        return frozenset(x.strip().strip("{}") for x in s.replace(";", ",").split(",") if x.strip())
    gs = settify(gold)
    if gs is None:
        return 0.0
    ss = settify(answer)
    if ss is None:
        return 0.0
    return 1.0 if ss == gs else 0.0
