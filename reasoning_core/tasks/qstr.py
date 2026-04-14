import random
from dataclasses import dataclass
from itertools import product
from collections import defaultdict, deque

from reasoning_core.template import Task, Problem, Config, edict

# ---- Core types ---------------------------------------------------------

@dataclass(frozen=True)
class Calculus:
    name: str
    base: frozenset
    converse: dict
    compose: dict   # (r1, r2) -> frozenset(base)

# ---- Sound derivation: enumerate, verify saturation --------------------

def _build(domain, rel):
    dom = list(domain)
    n = len(dom)
    R = [[rel(dom[i], dom[j]) for j in range(n)] for i in range(n)]
    cv, cp, seen = {}, defaultdict(set), set()
    for i in range(n):
        Ri = R[i]
        for j in range(n):
            rij = Ri[j]
            cv[rij] = R[j][i]
            seen.add(rij)
            Rj = R[j]
            for k in range(n):
                cp[rij, Rj[k]].add(Ri[k])
    return seen, cv, cp

def derive(name, enum, rel, N):
    """Enumerate over enum(N); confirm fixpoint by rebuilding at enum(N+1).
    If the assertion fires, the table grew — bump N until it stabilises."""
    s1, _,   cp1 = _build(enum(N),     rel)
    s2, cv2, cp2 = _build(enum(N + 1), rel)
    assert s1 == s2,   f"{name}: base relations grew N={N}->{N+1}; bump N"
    assert cp1 == cp2, f"{name}: compose table grew N={N}->{N+1}; bump N"
    return Calculus(name, frozenset(s2), dict(cv2),
                    {k: frozenset(v) for k, v in cp2.items()})

def derive_product(name, calc):
    """Cartesian product of `calc` with itself. Sound when the two factor
    dimensions are independently realisable in the runtime domain (true for
    axis-aligned boxes: x and y can be chosen freely)."""
    base = frozenset((r1, r2) for r1 in calc.base for r2 in calc.base)
    cv = {(a, b): (calc.converse[a], calc.converse[b]) for (a, b) in base}
    cp = {}
    U = calc.base
    for (a1, b1) in base:
        for (a2, b2) in base:
            cp[(a1, b1), (a2, b2)] = frozenset(
                (x, y)
                for x in calc.compose.get((a1, a2), U)
                for y in calc.compose.get((b1, b2), U))
    return Calculus(name, base, cv, cp)

# ---- Path-consistency (sound; not complete on full Allen, which only
#      affects rejection rate, not the validity of accepted singletons) --

def closure(calc, n, hard):
    U = calc.base
    R = {(i, j): U for i in range(n) for j in range(n) if i != j}
    for (i, j), s in hard.items():
        R[i, j] = R[i, j] & s
        R[j, i] = R[j, i] & frozenset(calc.converse[r] for r in s)
        if not R[i, j]:
            return None
    changed = True
    while changed:
        changed = False
        for i, j, k in product(range(n), repeat=3):
            if i == j or j == k or i == k:
                continue
            comp = frozenset().union(*(
                calc.compose.get((r1, r2), U)
                for r1 in R[i, j] for r2 in R[j, k]))
            new = R[i, k] & comp
            if not new:
                return None
            if new != R[i, k]:
                R[i, k] = new
                changed = True
    return R

# ---- Concrete semantics: intervals on [-N, N] (symmetric => nesting) ---

def all_intervals(N):
    pts = range(-N, N + 1)
    return [(a, b) for a in pts for b in pts if a < b]

def all_boxes(N):
    iv = all_intervals(N)
    return [(x0, x1, y0, y1) for (x0, x1) in iv for (y0, y1) in iv]

def allen(a, b):
    sgn = lambda x, y: (x > y) - (x < y)
    return (sgn(a[0], b[0]), sgn(a[0], b[1]),
            sgn(a[1], b[0]), sgn(a[1], b[1]))

def hdir(a, b): return allen((a[0], a[1]), (b[0], b[1]))
def vdir(a, b): return allen((a[2], a[3]), (b[2], b[3]))

def rcc8_iv(a, b):
    """RCC8 on 1D intervals — used for derivation. The composition table is
    dimension-independent, so this table holds for 2D boxes at runtime."""
    a0, a1 = a; b0, b1 = b
    if a1 <  b0 or b1 <  a0:                              return 'DC'
    if a1 == b0 or b1 == a0:                              return 'EC'
    if a == b:                                            return 'EQ'
    if b0 < a0 and a1 < b1:                               return 'NTPP'
    if a0 < b0 and b1 < a1:                               return 'NTPPi'
    if (a0 == b0 and a1 < b1) or (b0 < a0 and a1 == b1): return 'TPP'
    if (a0 == b0 and b1 < a1) or (a0 < b0 and a1 == b1): return 'TPPi'
    return 'PO'

def rcc8_box(a, b):
    """RCC8 on 2D axis-aligned boxes — runtime ground truth."""
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    cx_lo, cx_hi = max(ax0, bx0), min(ax1, bx1)
    cy_lo, cy_hi = max(ay0, by0), min(ay1, by1)
    if cx_lo >  cx_hi or cy_lo >  cy_hi: return 'DC'
    if cx_lo == cx_hi or cy_lo == cy_hi: return 'EC'
    if a == b: return 'EQ'
    a_in = bx0 <= ax0 and ax1 <= bx1 and by0 <= ay0 and ay1 <= by1
    b_in = ax0 <= bx0 and bx1 <= ax1 and ay0 <= by0 and by1 <= ay1
    tan  = (ax0 == bx0 or ax1 == bx1 or ay0 == by0 or ay1 == by1)
    if a_in: return 'TPP'  if tan else 'NTPP'
    if b_in: return 'TPPi' if tan else 'NTPPi'
    return 'PO'

def coarse_iv(a, b):
    if a[1] <  b[0]: return 'before'
    if b[1] <  a[0]: return 'after'
    return 'overlap'

def cardinal_box(a, b):
    return (coarse_iv((a[0], a[1]), (b[0], b[1])),
            coarse_iv((a[2], a[3]), (b[2], b[3])))

# ---- Labels -------------------------------------------------------------

ALLEN_NAMES = {
    (-1,-1,-1,-1): 'before',       (-1,-1, 0,-1): 'meets',
    (-1,-1, 1,-1): 'overlaps',     (-1,-1, 1, 0): 'finished-by',
    (-1,-1, 1, 1): 'contains',     ( 0,-1, 1,-1): 'starts',
    ( 0,-1, 1, 0): 'equals',       ( 0,-1, 1, 1): 'started-by',
    ( 1,-1, 1,-1): 'during',       ( 1,-1, 1, 0): 'finishes',
    ( 1,-1, 1, 1): 'overlapped-by',( 1, 0, 1, 1): 'met-by',
    ( 1, 1, 1, 1): 'after',
}
CARDINAL_NAMES = {
    ('before', 'before'):  'south-west',
    ('before', 'overlap'): 'west',
    ('before', 'after'):   'north-west',
    ('overlap','before'):  'south',
    ('overlap','overlap'): 'overlapping',
    ('overlap','after'):   'north',
    ('after',  'before'):  'south-east',
    ('after',  'overlap'): 'east',
    ('after',  'after'):   'north-east',
}

RCC8_NAMES = {
    'DC':    'disconnected-from',
    'EC':    'touches',
    'PO':    'partially-overlaps',
    'EQ':    'equals',
    'TPP':   'tangential-part-of',
    'NTPP':  'non-tangential-part-of',
    'TPPi':  'has-tangential-part',
    'NTPPi': 'has-non-tangential-part',
}

_id = lambda r: r

# ---- Build calculi (all under a second of module load) -----------------

ALLEN_CALC    = derive('allen',  all_intervals, allen,     N=4)
RCC8_CALC     = derive('rcc8',   all_intervals, rcc8_iv,   N=3)
COARSE_CALC   = derive('coarse', all_intervals, coarse_iv, N=3)
CARDINAL_CALC = derive_product('cardinal', COARSE_CALC)

assert len(ALLEN_CALC.base)    == 13
assert len(RCC8_CALC.base)     == 8
assert len(COARSE_CALC.base)   == 3
assert len(CARDINAL_CALC.base) == 9

# ---- Registry: shared calculi, calculus-specific runtime semantics -----

_INT_POOL = all_intervals(4)
_BOX_POOL = all_boxes(3)

REGISTRY = {
    'allen_time': dict(calc=ALLEN_CALC, pool=_INT_POOL, rel=allen,
                       label=ALLEN_NAMES.__getitem__,
                       topic='time intervals',
                       phrasing='the temporal relation of interval {i} to interval {j}'),
    'allen_x':    dict(calc=ALLEN_CALC, pool=_BOX_POOL, rel=hdir,
                       label=ALLEN_NAMES.__getitem__,
                       topic='horizontal extents of 2D boxes',
                       phrasing='the relation of the horizontal extent of box {i} to that of box {j}'),
    'allen_y':    dict(calc=ALLEN_CALC, pool=_BOX_POOL, rel=vdir,
                       label=ALLEN_NAMES.__getitem__,
                       topic='vertical extents of 2D boxes',
                       phrasing='the relation of the vertical extent of box {i} to that of box {j}'),
    'rcc8': dict(calc=RCC8_CALC, pool=_BOX_POOL, rel=rcc8_box,
                 label=RCC8_NAMES.__getitem__,
                 topic='2D regions (axis-aligned boxes)',
                 phrasing='the spatial relation of region {i} to region {j}'),
    'cardinal':   dict(calc=CARDINAL_CALC, pool=_BOX_POOL, rel=cardinal_box,
                       label=CARDINAL_NAMES.__getitem__,
                       topic='2D boxes by cardinal direction',
                       phrasing='the cardinal direction of box {i} relative to box {j}'),
}

# ---- Tree utilities -----------------------------------------------------

def random_tree(n):
    nodes = list(range(n)); random.shuffle(nodes)
    return [(nodes[i], random.choice(nodes[:i])) for i in range(1, n)]

def farthest_pair(n, edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    def bfs(src):
        dist = {src: 0}; q = deque([src]); far = src
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    if dist[v] > dist[far]: far = v
                    q.append(v)
        return far, dist[far]
    a, _ = bfs(0)
    b, d = bfs(a)
    return a, b, d

# ---- Task ---------------------------------------------------------------

@dataclass
class QualitativeReasoningConfig(Config):
    n_entities: int = 5
    extra_edges: int = 0

    def update(self, c=1):
        self.n_entities += c
        self.extra_edges += 0.5 * c

class QualitativeReasoning(Task):
    def __init__(self, config=None):
        super().__init__(config=config or QualitativeReasoningConfig())

    def generate(self):
        for _ in range(80):
            p = self._try()
            if p is not None:
                return p
        raise RuntimeError("QualitativeReasoning: could not build instance")

    def _try(self):
        cfg = self.config
        key = random.choice(list(REGISTRY))
        spec = REGISTRY[key]
        calc, pool, rel = spec['calc'], spec['pool'], spec['rel']
        label, topic, phrasing = spec['label'], spec['topic'], spec['phrasing']
        n = max(3, int(cfg.n_entities))

        ents = [random.choice(pool) for _ in range(n)]
        gt = {(i, j): rel(ents[i], ents[j])
              for i in range(n) for j in range(n) if i != j}

        tree = random_tree(n)
        qi, qj, hops = farthest_pair(n, tree)
        if hops < 2:
            return None

        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        tree_set  = {frozenset(e) for e in tree}
        query_set = frozenset({qi, qj})
        extras_pool = [e for e in all_pairs
                       if frozenset(e) not in tree_set
                       and frozenset(e) != query_set]
        random.shuffle(extras_pool)

        n_extra = max(0, int(cfg.extra_edges))
        revealed = list(tree) + extras_pool[:n_extra]
        leftover = extras_pool[n_extra:]

        hard = {e: frozenset({gt[e]}) for e in revealed}
        R = closure(calc, n, hard)
        while R is not None and len(R[qi, qj]) > 1 and leftover:
            e = leftover.pop()
            hard[e] = frozenset({gt[e]})
            revealed.append(e)
            R = closure(calc, n, hard)

        if R is None or len(R[qi, qj]) != 1:
            return None
        truth = next(iter(R[qi, qj]))
        if truth != gt[qi, qj]:
            return None  # impossible if PC is sound; defensive

        vocab = sorted({label(r) for r in calc.base})
        metadata = edict(
            calculus=key, topic=topic, phrasing=phrasing,
            n_entities=n, hops=hops, n_revealed=len(revealed),
            entities=ents,
            revealed=[(i, j, label(gt[i, j])) for (i, j) in revealed],
            query=(qi, qj), vocabulary=vocab,
        )
        return Problem(metadata=metadata, answer=label(gt[qi, qj]))

    def prompt(self, metadata):
        qi, qj = metadata.query
        lines = [
            f"Qualitative reasoning over {metadata.topic}.",
            f"There are {metadata.n_entities} entities labeled 0 through {metadata.n_entities - 1}.",
            "You are given the following facts (read 'i rel j' as 'entity i is rel to entity j'):",
        ]
        lines += [f"  {i} {r} {j}" for i, j, r in metadata.revealed]
        lines += [
            "",
            f"Question: what is {metadata.phrasing.format(i=qi, j=qj)}?",
            f"The answer is exactly one of: {', '.join(metadata.vocabulary)}.",
            "Respond with only the relation name as the answer.",
        ]
        return "\n".join(lines)
        
    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        norm = lambda s: str(s).strip().lower().replace('_', '-')
        return float(norm(answer) == norm(entry.answer))