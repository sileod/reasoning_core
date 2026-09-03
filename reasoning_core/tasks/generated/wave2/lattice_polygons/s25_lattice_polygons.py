import math
import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict
from reasoning_core.utils import score_scalar

TASK_META = {'parent_source_id': None,
 'idea': 'Add measure reasoning over lattice polygons.',
 'hypothesis': 'S25',
 'changes': 'Ask for the area, the boundary point count, or the interior point '
            'count of a stated polygon.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3944590077,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def _shoelace_area2(verts):
    s = 0
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s)


def _boundary_points(verts):
    n = len(verts)
    b = 0
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        b += _gcd(x2 - x1, y2 - y1)
    return b


def _segments_intersect(p1, p2, p3, p4):
    def cross(ax, ay, bx, by):
        return ax * by - ay * bx

    def on_segment(ax, ay, bx, by, cx, cy):
        return min(ax, bx) <= cx <= max(ax, bx) and min(ay, by) <= cy <= max(ay, by)

    d1 = cross(p3[0] - p1[0], p3[1] - p1[1], p2[0] - p1[0], p2[1] - p1[1])
    d2 = cross(p4[0] - p1[0], p4[1] - p1[1], p2[0] - p1[0], p2[1] - p1[1])
    d3 = cross(p1[0] - p3[0], p1[1] - p3[1], p4[0] - p3[0], p4[1] - p3[1])
    d4 = cross(p2[0] - p3[0], p2[1] - p3[1], p4[0] - p3[0], p4[1] - p3[1])

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    if d1 == 0 and on_segment(p3[0], p3[1], p4[0], p4[1], p1[0], p1[1]):
        return True
    if d2 == 0 and on_segment(p3[0], p3[1], p4[0], p4[1], p2[0], p2[1]):
        return True
    if d3 == 0 and on_segment(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]):
        return True
    if d4 == 0 and on_segment(p1[0], p1[1], p2[0], p2[1], p4[0], p4[1]):
        return True
    return False


def _is_simple(verts):
    n = len(verts)
    for i in range(n):
        for j in range(i + 1, n):
            if j == i or j == (i + 1) % n or i == (j + 1) % n:
                continue
            if _segments_intersect(verts[i], verts[(i + 1) % n], verts[j], verts[(j + 1) % n]):
                return False
    return True


@dataclass
class LatticePolygonsConfig(Config):
    n_verts: int = 5
    coord_range: int = 8

    def apply_difficulty(self, level):
        self.n_verts = int(self.n_verts + level)
        self.coord_range = int(self.coord_range + 3 * level)


class LatticePolygons(Task):
    config_cls = LatticePolygonsConfig

    def generate_entry(self):
        n = self.config.n_verts
        rang = self.config.coord_range

        verts = None
        for _ in range(400):
            pts = {(random.randint(-rang, rang), random.randint(-rang, rang))
                   for _ in range(n)}
            if len(pts) < n:
                continue
            pts = list(pts)
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            order = sorted(pts, key=lambda p: (math.atan2(p[1] - cy, p[0] - cx)))
            if _is_simple(order):
                verts = order
                break

        if verts is None:
            raise RuntimeError("could not build a simple polygon")

        area2 = _shoelace_area2(verts)
        boundary = _boundary_points(verts)
        interior = (area2 - boundary) // 2 + 1

        kinds = ['area', 'boundary', 'interior']
        kind = random.choice(kinds)
        if kind == 'area':
            q = "twice its area (the shoelace value, an integer)"
            answer = area2
        elif kind == 'boundary':
            q = "the number of lattice points on its boundary"
            answer = boundary
        else:
            q = "the number of lattice points strictly inside it"
            answer = interior

        assert isinstance(answer, int)
        assert answer >= 0

        metadata = edict({
            'vertices': [[int(x), int(y)] for x, y in verts],
            'kind': kind,
            'area2': int(area2),
            'boundary': int(boundary),
            'interior': int(interior),
        })
        metadata.payload = {
            'vertices': metadata.vertices,
            'kind': kind,
            'question': q,
        }
        return Entry(metadata=metadata, answer=str(answer))

    def render_prompt(self, metadata):
        verts_str = ', '.join(f'({x},{y})' for x, y in metadata.vertices)
        return (
            f"A simple polygon on the integer grid has vertices, in order:\n"
            f"{verts_str}\n\n"
            f"What is {metadata.payload.question}? The answer is an integer."
        )

    def score_answer(self, answer, entry):
        return score_scalar(answer, entry)


def generate_samples():
    random.seed(3944590077)
    task = LatticePolygons()
    out = []
    for level in [0, 2, 5]:
        task.config.set_level(level)
        out.append(f"## Level {level}\n")
        for _ in range(2):
            ex = task.generate_example()
            out.append("**Prompt:**\n" + ex.metadata.prompt + "\n")
            out.append("**Answer:**\n" + ex.answer + "\n")
    return "\n".join(out)
