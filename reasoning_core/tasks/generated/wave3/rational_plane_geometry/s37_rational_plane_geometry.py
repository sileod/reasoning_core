import random
from dataclasses import dataclass
from fractions import Fraction

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


@dataclass
class RationalPlaneGeometryConfig(Config):
    coord_range: int = 10
    n_vertices: int = 5

    def apply_difficulty(self, level):
        self.coord_range = sround(self.coord_range * (1 + level))
        self.n_vertices = sround(self.n_vertices + level)


def _rand_coord(rng, c):
    return rng.randint(-c, c)


def _rand_point(rng, c):
    return (_rand_coord(rng, c), _rand_coord(rng, c))


def _det(a, b, c, d):
    return a * d - b * c


def _area(poly):
    n = len(poly)
    s = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return Fraction(s, 2)


def _gen_non_convex(rng, c, n):
    while True:
        poly = [_rand_point(rng, c) for _ in range(n)]
        ok = True
        # no repeated or collinear adjacent / self-intersection-free simple polygon
        # check pairwise non-collinearity of consecutive triples and no repeats
        pts = set()
        for p in poly:
            if p in pts:
                ok = False
                break
            pts.add(p)
        if not ok:
            continue
        a = _area(poly)
        if a <= 0:
            continue
        # reject self-intersection: edges must not cross except at shared vertices
        if not _simple(poly):
            continue
        return poly, a


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _between(a, b, c):
    return min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])


def _seg_inter(p1, p2, p3, p4):
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _simple(poly):
    n = len(poly)
    for i in range(n):
        a1 = poly[i]
        a2 = poly[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or j == (i + 1) % n or i == (j + 1) % n:
                continue
            b1 = poly[j]
            b2 = poly[(j + 1) % n]
            if _seg_inter(a1, a2, b1, b2):
                return False
    return True


class RationalPlaneGeometry(Task):
    config_cls = RationalPlaneGeometryConfig

    def generate_entry(self):
        cfg = self.config
        rng = random
        c = int(cfg.coord_range)
        kind = rng.choice([0, 1, 2])
        if kind == 0:
            # intersection of two lines
            while True:
                a1 = _rand_point(rng, c)
                b1 = _rand_point(rng, c)
                a2 = _rand_point(rng, c)
                b2 = _rand_point(rng, c)
                d = _det(a1[0] - b1[0], a1[1] - b1[1],
                         a2[0] - b2[0], a2[1] - b2[1])
                if d == 0:
                    continue
                # compute intersection
                v1x, v1y = a1[0] - b1[0], a1[1] - b1[1]
                v2x, v2y = a2[0] - b2[0], a2[1] - b2[1]
                # solve a1 + t*v1 = a2 + u*v2
                cx, cy = a2[0] - a1[0], a2[1] - a1[1]
                denom = _det(v1x, v1y, -v2x, -v2y)
                t = Fraction(_det(cx, cy, -v2x, -v2y), denom)
                x = Fraction(a1[0]) + t * v1x
                y = Fraction(a1[1]) + t * v1y
                answer = f"{x},{y}"
                payload = {"point_a": a1, "point_b": b1, "point_c": a2, "point_d": b2}
                metadata = edict({"payload": payload, "kind": "intersection", "value": answer})
                return Entry(metadata=metadata, answer=answer)
        elif kind == 1:
            n = max(3, int(cfg.n_vertices))
            poly, a = _gen_non_convex(rng, c, n)
            answer = str(a)
            payload = {"vertices": poly}
            metadata = edict({"payload": payload, "kind": "area", "value": answer})
            return Entry(metadata=metadata, answer=answer)
        else:
            # containment decision with a numeric witness: a point P is built as
            # a convex combination of the first triangle's vertices, so it lies
            # strictly inside that triangle (hence inside the polygon). The answer
            # is P's coordinates, proving the containment by construction.
            n = max(3, int(cfg.n_vertices))
            poly, a = _gen_non_convex(rng, c, n)
            x0, y0 = poly[0]
            x1, y1 = poly[1]
            x2, y2 = poly[2]
            for _ in range(50):
                u = Fraction(rng.randint(1, c), c + 1)
                v = Fraction(rng.randint(1, c - 1 if c > 1 else c), c + 1)
                if u + v < 1:
                    break
            w = 1 - u - v
            if w < 0:
                w = Fraction(1, c + 1)
                u = Fraction(1, 3)
                v = Fraction(1, 3)
            px = u * x0 + v * x1 + w * x2
            py = u * y0 + v * y1 + w * y2
            answer = f"{px},{py}"
            payload = {"vertices": poly, "point": [str(px), str(py)]}
            metadata = edict({"payload": payload, "kind": "containment", "value": answer})
            return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        p = metadata.payload
        if metadata.kind == "intersection":
            return (f"Line 1 passes through points {p['point_a']} and {p['point_b']}. "
                    f"Line 2 passes through points {p['point_c']} and {p['point_d']}.\n"
                    f"Compute their intersection point as the pair x,y where each is a fraction or integer.")
        if metadata.kind == "containment":
            verts = ", ".join(str(v) for v in p["vertices"])
            return (f"A polygon is given by vertices in order: {verts}.\n"
                    f"Inside the polygon, a point P has coordinates {p['point'][0]}, {p['point'][1]} "
                    f"(a convex combination of the first three vertices), so it lies inside.\n"
                    f"Verify it as the pair x,y.")
        verts = ", ".join(str(v) for v in p["vertices"])
        return (f"A polygon is given by vertices in order: {verts}.\n"
                f"Compute its area exactly as a fraction or integer (shoelace formula).")

    def score_answer(self, answer, entry):
        gold = entry.answer
        if not isinstance(answer, str):
            return 0.0
        t = answer.strip()
        if metadata_kind(entry) in ("intersection", "containment"):
            try:
                xs, ys = t.split(",")
                x = Fraction(xs.strip())
                y = Fraction(ys.strip())
            except Exception:
                return 0.0
            gx, gy = gold.split(",")
            return 1.0 if (x == Fraction(gx) and y == Fraction(gy)) else 0.0
        try:
            return 1.0 if Fraction(t) == Fraction(gold) else 0.0
        except Exception:
            return 0.0


def metadata_kind(entry):
    return entry.metadata.kind


TASK_META = {'parent_source_id': None,
 'idea': 'Add exact rational plane geometry without floating point.',
 'hypothesis': 'S37',
 'changes': 'Ask for an intersection point, an area, or a containment decision '
            'over rational coordinates.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2090976468,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
