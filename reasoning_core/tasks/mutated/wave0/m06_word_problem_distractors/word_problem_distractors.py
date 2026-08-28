"""Word-problem mutation: formally irrelevant but query-coupled distractor chains.

Derives from the relational family of ``reasoning_core/tasks/arithmetics.py``
(:class:`MathWordProblem`).  Alters the distribution to attach "distractor
chains": extra people related to a core person through the *same entity and
unit*, so they superficially look coupled to the query even though they are
formally irrelevant to determining the asked value.  The minimal relational
proof depth for the query is held fixed (equal to the core's own depth), and
irrelevance is verified formally with sympy linear algebra.
"""

import random
import sympy as sp
from itertools import combinations
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround
from reasoning_core.utils import score_scalar

UNITS = "stamps marbles coins books apples cards tokens beads tiles cookies shells stickers pebbles buttons".split()
NAMES = ("Mara Jon Aisha Wei Sofia Diego Priya Tom Lena Omar Yuki Hana Carlos Nina "
         "Zara Ravi Mei Iris Noah Amara Leo Sana Kof Tara").split()
ORD = {2: "half", 3: "a third", 4: "a quarter", 5: "a fifth"}
DIST_UNITS = "ribbons buttons cards beads shells pens".split()

TASK_META = {'parent_source_id': 'c267a83e5953e4862bec61fb7c72a249dc6d8d945f1116585ac947e52ef26f35',
 'idea': 'Test formally irrelevant but query-coupled distractors.',
 'hypothesis': 'H2',
 'changes': 'Add connected distractor chains sharing entities and units while '
            'preserving the proof core.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2364728918,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 20,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class WordProblemDistractorConfig(Config):
    n_rel: int = 2
    max_n: int = 12
    min_core_depth: int = 2
    distractor_p: float = 0.9
    chain_max: int = 2

    def apply_difficulty(self, level):
        self.n_rel = sround(self.n_rel + level)
        self.max_n = sround(self.max_n + 12 * level)
        self.min_core_depth = sround(self.min_core_depth + (level > 0))
        self.chain_max = sround(self.chain_max + (level > 3))


def ri(a, b):
    return random.randint(int(a), int(b))


def clean_answer(a):
    return isinstance(a, int) and 0 < a < 6000


def relation_text(rel, unit):
    op, a, b, k, c = rel
    if op == "times":
        return f"{a} has {k} times as many {unit} as {b}"
    if op == "more":
        return f"{a} has {k} more {unit} than {b}"
    if op == "fewer":
        return f"{a} has {k} fewer {unit} than {b}"
    if op == "frac":
        return f"{a} has {ORD[k]} as many {unit} as {b}"
    return f"{a} has as many {unit} as {b} and {c} combined"


def proof_core_size(names, rels, given, asked, given_value):
    """Minimum relation equations that identify ``asked`` from ``given``."""
    symbols = {n: sp.Symbol(n) for n in names}
    equations = []
    for op, a, b, k, c in rels:
        rhs = symbols[b] + symbols[c] if op == "combine" else {
            "times": k * symbols[b],
            "more": symbols[b] + k,
            "fewer": symbols[b] - k,
            "frac": symbols[b] / k,
        }[op]
        equations.append(sp.Eq(symbols[a], rhs))
    variables = [symbols[n] for n in names]
    asked_index = names.index(asked)
    given_eq = sp.Eq(symbols[given], given_value)
    for size in range(len(equations) + 1):
        for subset in combinations(equations, size):
            matrix, _ = sp.linear_eq_to_matrix((given_eq, *subset), variables)
            if all(vector[asked_index] == 0 for vector in matrix.nullspace()):
                return size
    return None


def build_core(config):
    """Build a plain relational core (people/relation/values), as the parent."""
    unit = random.choice(UNITS)
    m = ri(3, min(6, 3 + config.n_rel))
    names = random.sample(NAMES, m)
    x = sp.Symbol("x", positive=True)
    val, rels = {names[0]: x}, []
    for i in range(1, m):
        a = names[i]
        parents = names[:i]
        ops = ["times", "more", "fewer", "frac"] + (["combine"] if len(parents) >= 2 else [])
        op = random.choice(ops)
        if op == "times":
            b, k = random.choice(parents), ri(2, 4)
            val[a] = k * val[b]
            rels.append((op, a, b, k, None))
        elif op == "more":
            b, k = random.choice(parents), ri(2, config.max_n)
            val[a] = val[b] + k
            rels.append((op, a, b, k, None))
        elif op == "fewer":
            b, k = random.choice(parents), ri(2, config.max_n)
            val[a] = val[b] - k
            rels.append((op, a, b, k, None))
        elif op == "frac":
            b, k = random.choice(parents), random.choice([2, 3, 4])
            val[a] = val[b] / k
            rels.append((op, a, b, k, None))
        else:
            b, c = random.sample(parents, 2)
            val[a] = val[b] + val[c]
            rels.append((op, a, b, None, c))

    base = nums = None
    candidates = list(range(2, int(config.max_n) + 1))
    random.shuffle(candidates)
    for cand in candidates[:24]:
        cur = {k: sp.nsimplify(v.subs(x, cand)) for k, v in val.items()}
        if all(t.is_integer and t > 0 for t in cur.values()):
            base = cand
            nums = {k: int(v) for k, v in cur.items()}
            break
    if base is None:
        return None
    return dict(unit=unit, names=names, rels=rels, val=val, nums=nums, base=base)


def pick_query(core, config):
    """Pick (given, asked) with a proof core of at least min_core_depth."""
    names, rels, nums = core["names"], core["rels"], core["nums"]
    revealable = [z for z in names if nums[z] >= 2]
    if not revealable:
        return None
    pairs = []
    for given in revealable:
        for asked in names:
            if asked == given:
                continue
            size = proof_core_size(names, rels, given, asked, nums[given])
            if size is not None and size >= config.min_core_depth:
                pairs.append((given, asked, size))
    if not pairs:
        return None
    return random.choice(pairs)


def attach_distractors(core, given, asked, config):
    """Attach connected distractor chains that are formally irrelevant to asked."""
    names = list(core["names"])
    rels = list(core["rels"])
    nums = dict(core["nums"])
    # Pool of names not already used.
    pool = [n for n in NAMES if n not in names]
    extra_names = []
    dangling = [n for n in names if n not in (given, asked)]

    n_chains = ri(1, config.chain_max)
    for _ in range(n_chains):
        if random.random() > config.distractor_p or not pool:
            break
        anchor = random.choice(names)
        new = pool.pop()
        # Distractor is a *child* of a core person: depends on the core but
        # nothing in the core depends on it, so it never feeds the asked value.
        op = random.choice(["times", "more", "fewer", "frac"])
        if op == "times":
            k = ri(2, 4)
            nums[new] = int(k * nums[anchor])
            rels.append((op, new, anchor, k, None))
        elif op == "more":
            k = ri(2, config.max_n)
            nums[new] = int(nums[anchor] + k)
            rels.append((op, new, anchor, k, None))
        elif op == "fewer":
            k = ri(2, max(2, nums[anchor] - 1))
            if nums[anchor] - k < 1:
                break
            nums[new] = int(nums[anchor] - k)
            rels.append((op, new, anchor, k, None))
        else:
            k = random.choice([2, 3, 4])
            if nums[anchor] % k != 0:
                break
            nums[new] = int(nums[anchor] / k)
            rels.append((op, new, anchor, k, None))
        extra_names.append(new)
        names = names + [new]
        anchor = new
    return names, rels, nums, extra_names


def generate_entry(config):
    for _ in range(200):
        core = build_core(config)
        if core is None:
            continue
        query = pick_query(core, config)
        if query is None:
            continue
        given, asked, core_size = query
        names, rels, nums, extra = attach_distractors(core, given, asked, config)
        if not extra:
            continue
        # Formal irrelevance check: the proof core must be unchanged by the
        # distractor relations (the asked value from given needs no distractor).
        full_size = proof_core_size(names, rels, given, asked, nums[given])
        if full_size is not None and full_size == core_size and int(nums[asked]) > 0:
            random.shuffle(rels)
            metadata = edict(
                family="relational",
                unit=core["unit"],
                names=names,
                relations=rels,
                given=given,
                asked=asked,
                given_value=nums[given],
                values=nums,
                distractor_names=extra,
                distractor_count=len(extra),
                proof_core_size=core_size,
                equation=str(sp.Eq(core["val"][given], nums[given])),
                cot="\n".join(relation_text(r, core["unit"]) for r in rels),
            )
            return Entry(metadata=metadata, answer=str(nums[asked]))
    raise RuntimeError("no valid word-problem distractor instance generated")


class WordProblemDistractors(Task):
    summary = "Relational word problems whose asked value is unchanged by attached distractor chains."
    config_cls = WordProblemDistractorConfig

    def generate_entry(self):
        return generate_entry(self.config)

    def render_prompt(self, m):
        lines = ". ".join(relation_text(r, m.unit) for r in m.relations)
        return (
            f"{lines}. {m.given} has {m.given_value} {m.unit}. "
            f"How many {m.unit} does {m.asked} have? Answer with a number."
        )

    def score_answer(self, answer, entry):
        return score_scalar(answer, entry)

    def balancing_key(self, problem):
        m = problem.metadata
        ops = ",".join(r[0] for r in m.relations)
        return f"relational:d{min(m.proof_core_size, 3)}:{ops}:x{m.distractor_count}"

    def deduplication_key(self, problem):
        m = problem.metadata
        return str((m.unit, tuple(map(tuple, m.relations)), m.given, m.given_value, m.asked))
