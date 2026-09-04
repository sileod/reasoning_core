import random
import math
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'coordinate_frame_composition (draw 1 of 1)',
 'hypothesis': 'HV-065',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/coordinate_frame_composition',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2972124136,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


ROTATION = {
    0: ((1, 0), (0, 1)),
    90: ((0, -1), (1, 0)),
    180: ((-1, 0), (0, -1)),
    270: ((0, 1), (-1, 0)),
}

# Reflection matrices acting as (x, y) -> M (x, y)
REFLECTION_VERT = ((-1, 0), (0, 1))   # across the vertical axis
REFLECTION_HORIZ = ((1, 0), (0, -1))  # across the horizontal axis


def _mat_mul(A, B):
    return (
        (A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]),
        (A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]),
    )


def _mat_vec(M, v):
    return (M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1])


def link_matrix(angle, reflect):
    M = ROTATION[angle]
    if reflect is not None:
        M = _mat_mul(reflect, M)
    return M


def compose(A, B):
    """Affine turns: B after A. Each is (M, t) with f(p) = M p + t."""
    (AM, At), (BM, Bt) = A, B
    M = _mat_mul(BM, AM)
    t = tuple(a + b for a, b in zip(_mat_vec(BM, At), Bt))
    return (M, t)


def invert(T):
    M, t = T
    # M is an integer orthogonal matrix (det = +-1), so M^{-1} = M^T (integer).
    MT = ((M[0][0], M[1][0]), (M[0][1], M[1][1]))
    nt = _mat_vec(MT, (-t[0], -t[1]))
    return (MT, nt)


def apply_affine(T, v):
    M, t = T
    return tuple(a + b for a, b in zip(_mat_vec(M, v), t))


@dataclass
class CoordinateFrameCompositionConfig(Config):
    frames: int = 5
    coord_max: int = 6

    def apply_difficulty(self, level):
        self.frames = sround(self.frames + level)
        self.coord_max = sround(self.coord_max + 4 * level)


class CoordinateFrameComposition(Task):
    summary = "Compose discrete translations, rotations, and reflections between named coordinate frames, returning a queried point or vector in the target frame."
    config_cls = CoordinateFrameCompositionConfig

    def generate_entry(self):
        n = self.config.frames
        coord_max = self.config.coord_max
        frames = list(range(n))

        links = []  # links[k] maps frame k -> frame k+1 as (angle, reflect_label, dx, dy)
        for _ in range(n - 1):
            angle = random.choice([0, 90, 180, 270])
            reflect = random.choice([None, "vertical", "horizontal"])
            dx = random.randint(-coord_max, coord_max)
            dy = random.randint(-coord_max, coord_max)
            links.append((angle, reflect, dx, dy))

        q, t = random.sample(frames, 2)
        px = random.randint(-coord_max, coord_max)
        py = random.randint(-coord_max, coord_max)

        reflect_mats = {"vertical": REFLECTION_VERT, "horizontal": REFLECTION_HORIZ}
        affine_links = [(link_matrix(a, reflect_mats[r] if r else None), (dx, dy))
                        for (a, r, dx, dy) in links]

        total = (( (1, 0), (0, 1) ), (0, 0))
        if q < t:
            for k in range(q, t):
                total = compose(total, affine_links[k])
        else:
            for k in range(t, q):
                total = compose(invert(affine_links[k]), total)

        ans = apply_affine(total, (px, py))
        assert isinstance(ans[0], int) and isinstance(ans[1], int)

        frame_names = [chr(ord("A") + i) for i in range(n)]
        payload_frames = []
        for k in range(n - 1):
            angle, reflect, dx, dy = links[k]
            src = frame_names[k]
            dst = frame_names[k + 1]
            payload_frames.append(
                "frame {} from frame {}: rotate {} degree(s) counterclockwise, ".format(dst, src, angle)
                + ("then reflect across the {} axis, ".format(reflect) if reflect else "")
                + "then translate by ({}, {})".format(dx, dy)
            )

        payload = edict({
            "point_location": "A point P has coordinates ({}, {}) in frame {}.".format(px, py, frame_names[q]),
            "relations": "\n".join("- " + s for s in payload_frames),
            "question": "What are the coordinates of P in frame {}?".format(frame_names[t]),
        })
        metadata = edict({
            "payload": payload,
            "point": (px, py),
            "source": q,
            "target": t,
            "n_frames": n,
        })
        metadata.answer = "({}, {})".format(ans[0], ans[1])
        return Entry(metadata=metadata, answer=metadata.answer)

    def render_prompt(self, metadata):
        preamble = (
            "Coordinate frames are related by discrete rigid transforms applied to coordinates in the "
            "source frame to obtain coordinates in the target frame, in this fixed order: first rotate "
            "counterclockwise by the given amount, then reflect across the named axis, then translate. "
            "A reflection across the vertical axis maps (x, y) to (-x, y); a reflection across the "
            "horizontal axis maps (x, y) to (x, -y). A counterclockwise rotation by theta maps a vector "
            "to (x*cos(theta) - y*sin(theta), x*sin(theta) + y*cos(theta))."
        )
        return "{}\n\n{}\n\nGive the answer as (x, y).".format(preamble, render_payload(metadata.payload))

    def score_answer(self, answer, entry):
        gold = entry.answer
        if str(answer).strip() == str(gold):
            return 1.0
        a = _parse_pair(answer)
        g = _parse_pair(gold)
        if a is not None and g is not None and a == g:
            return 1.0
        return 0.0


def _parse_pair(text):
    import re
    m = re.search(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", str(text))
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))
