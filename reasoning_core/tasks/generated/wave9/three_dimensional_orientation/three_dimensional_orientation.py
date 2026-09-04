"""Track labeled axes or faces through sequences of discrete 3D rotations."""

import math
import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'three_dimensional_orientation (draw 1 of 1)',
 'hypothesis': 'HV-066',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/three_dimensional_orientation',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2939315147,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

# 3D rotation matrices about each of the six cardinal axes (each a 90-degree turn).
# We use the standard right-handed coordinate frame with axes +X, -X, +Y, -Y, +Z, -Z.
# A rotation matrix R maps a point p to R p.
ROTATIONS = {}

def _rotation_matrix(axis_sign, axis):
    # axis in {"X","Y","Z"}, axis_sign in {1, -1}
    # right-handed rotation by 90 deg about the given axis
    c, s = 0, axis_sign
    mats = {
        "X": [[1, 0, 0], [0, c, -s], [0, s, c]],
        "Y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        "Z": [[c, -s, 0], [s, c, 0], [0, 0, 1]],
    }
    return mats[axis]

for ax in "XYZ":
    for sg in (1, -1):
        ROTATIONS[(sg, ax)] = _rotation_matrix(sg, ax)

def _mat_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def _mat_apply(M, v):
    return [sum(M[i][k] * v[k] for k in range(3)) for i in range(3)]

def _mat_equal(A, B):
    return all(abs(A[i][j] - B[i][j]) < 1e-9 for i in range(3) for j in range(3))

# Base directions; +X is (1,0,0) etc. Identity matrix maps each direction to itself.
IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# The set of 24 orientation-preserving rotations of the cube (proper rotations,
# determinant +1). Build by composing 90-degree turns.
def _all_rotations():
    rots = []
    keys = list(ROTATIONS.keys())
    for a in keys:
        for b in keys:
            for c in keys:
                M = _mat_mul(ROTATIONS[a], _mat_mul(ROTATIONS[b], ROTATIONS[c]))
                if not any(_mat_equal(M, R) for R in rots):
                    rots.append(M)
    return rots

def _is_proper(M):
    # determinant
    d = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
         - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
         + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    return abs(d - 1) < 1e-9

ALL_ROTATIONS = [M for M in _all_rotations() if _is_proper(M)]

# The six labeled faces/axes as directions from the center to the face center.
FACE_NAMES = {
    (1, 0, 0): "+X",
    (-1, 0, 0): "-X",
    (0, 1, 0): "+Y",
    (0, -1, 0): "-Y",
    (0, 0, 1): "+Z",
    (0, 0, -1): "-Z",
}
DIR_TO_FACE = {v: k for k, v in FACE_NAMES.items()}

COMPASS = ["+X", "+Y", "-X", "-Y"]  # east, north, west, south on the horizontal plane


class ThreeDimensionalOrientationConfig(Config):
    n_rotations: int = 6
    n_questions: int = 4

    def apply_difficulty(self, level):
        self.n_rotations = sround(3 + level)
        self.n_questions = sround(4 + (level // 2))


class ThreeDimensionalOrientation(Task):
    summary = (
        "Track labeled axes or faces through sequences of discrete 3D rotations, "
        "returning final orientation, adjacency, or relative direction."
    )
    config_cls = ThreeDimensionalOrientationConfig

    def __init__(self, config=None):
        super().__init__(config=config)
        self._rotation_pool = list(ROTATIONS.values())

    def _sample_rotation_mat(self, rng):
        M = _mat_mul(rng.choice(self._rotation_pool), rng.choice(self._rotation_pool))
        return M

    def _sample_rotation_label(self, rng):
        return rng.choice(list(ROTATIONS.keys()))

    def generate_entry(self):
        seed = random.randrange(2**32)
        rng = random.Random(seed)
        level = self.config.level

        n_rot = self.config.n_rotations
        n_quest = self.config.n_questions

        # Build a sequence of concrete 90-degree rotations about world axes.
        seq = []
        cumulative = IDENTITY
        for _ in range(n_rot):
            key = self._sample_rotation_label(rng)
            R = next(m for k, m in ROTATIONS.items() if k == key)
            seq.append(key)
            cumulative = _mat_mul(R, cumulative)

        # Ensure cumulative is a proper rotation with a nontrivial result (not identity)
        tries = 0
        while _mat_equal(cumulative, IDENTITY) and tries < 40:
            # repick the last rotation
            key = self._sample_rotation_label(rng)
            R = next(m for k, m in ROTATIONS.items() if k == key)
            seq[-1] = key
            cumulative = IDENTITY
            for k_ in seq:
                cumulative = _mat_mul(ROTATIONS[k_], cumulative)
            tries += 1

        # Instance: 'X' face color, sequence of rotations.
        # Question modes:
        #   mode A: after all rotations, what is the final orientation of a labeled face (+X, +Y, +Z)?
        #   mode B: adjacency - which labeled face is now opposite / adjacent to initial +X?
        #   mode C: relative direction - what world direction does a specified starting face now point?

        qs = []
        answers = []

        for i in range(n_quest):
            mode = rng.choice(["orientation", "relative", "opposite"])
            start_face = rng.choice(["+X", "+Y", "+Z"])
            if mode == "orientation":
                # final direction of the face normal, expressed as a global direction
                normal = FACE_NAMES[DIR_TO_FACE[start_face]]
                v = _mat_apply(cumulative, list(DIR_TO_FACE[start_face]))
                v = tuple(int(round(x)) for x in v)
                final_direction = FACE_NAMES[v]
                qs.append({"mode": "orientation", "face": start_face, "answer": final_direction})
                answers.append(final_direction)
            elif mode == "relative":
                # which initial face now points in the direction the +Z world axis pointed before
                # i.e. apply inverse rotation to world +Z
                v = _mat_apply(_inverse(cumulative), [0, 0, 1])
                v = tuple(int(round(x)) for x in v)
                initial_face = FACE_NAMES[v]
                qs.append({"mode": "relative", "question": "+Z", "answer": initial_face})
                answers.append(initial_face)
            else:
                # opposite: which labeled face has its normal antiparallel to the world +X
                # direction? Its initial normal n satisfies cumulative.n = (-1,0,0), so
                # n = inverse(cumulative).(-1,0,0).
                v = _mat_apply(_inverse(cumulative), [-1, 0, 0])
                v = tuple(int(round(x)) for x in v)
                initial_face = FACE_NAMES[v]
                qs.append({"mode": "opposite", "answer": initial_face})
                answers.append(initial_face)

        seq_labels = " ".join(f"{'-' if sg < 0 else ''}{ax}" for sg, ax in seq)

        metadata = edict({
            "seed": seed,
            "n_rotations": n_rot,
            "sequence": [[sg, ax] for sg, ax in seq],
            "sequence_label": seq_labels,
            "cumulative": cumulative,
            "questions": qs,
        })
        metadata.payload = {
            "sequence": seq_labels,
            "questions": [
                {"mode": q["mode"], "face": q.get("face", q.get("question"))}
                for q in qs
            ],
        }
        answer = " ; ".join(answers)
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = [
            "A cube is marked on three faces with colors A, B, C on the +X, +Y, and +Z faces.",
            "It is rotated in 90-degree steps around world axes (right-handed).",
            "Each rotation is written as the axis sign and letter, e.g. +X means a 90-degree",
            "counterclockwise turn about the +X world axis when viewed from the positive end.",
            "",
            f"Sequence of rotations: {metadata.payload['sequence']}",
            "",
        ]
        for q in metadata.payload["questions"]:
            mode = q["mode"]
            if mode == "orientation":
                lines.append(f"After the whole sequence, which global direction (e.g. +X, -Y, +Z) does the colored {q['face']} face now point toward?")
            elif mode == "relative":
                lines.append(f"Which original face (A=+X, B=+Y, C=+Z) now points in the direction the world +Z axis pointed before the rotations?")
            else:
                lines.append(f"After the whole sequence, which labeled face (A=+X, B=+Y, or C=+Z) now points opposite to the world +X direction?")
        lines.append("")
        lines.append("Give the answers as the global direction or original face label strings, one per question, separated by ' ; ' (space-semicolon-space).")
        lines.append("For example, if the two answers were +Z and B, write: +Z ; B")
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        gold = entry.answer
        # normalize
        def norm(s):
            return " ".join(s.split()).replace(" ; ", " ; ").strip()
        a = norm(answer)
        g = norm(gold)
        if a == g:
            return 1.0
        parts_a = [p.strip() for p in a.split(";")]
        parts_g = [p.strip() for p in g.split(";")]
        if len(parts_a) != len(parts_g):
            return 0.0
        correct = sum(1 for x, y in zip(parts_a, parts_g) if x == y)
        if correct == 0:
            return 0.0
        return correct / len(parts_g)


def _inverse(M):
    # rotation matrix: transpose
    return [[M[j][i] for j in range(3)] for i in range(3)]
