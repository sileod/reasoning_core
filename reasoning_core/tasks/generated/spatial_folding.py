import random
import re
from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, edict, stochastic_rounding as sround
from ._base import GeneratedMixin


FOLD_TEXT = {
    "left": "fold the left half over the right half",
    "right": "fold the right half over the left half",
    "top": "fold the top half over the bottom half",
    "bottom": "fold the bottom half over the top half",
}


@dataclass
class SpatialFoldingConfig(Config):
    n_folds: int = 1
    n_punches: int = 1

    def apply_difficulty(self, level):
        self.n_folds = sround(self.n_folds + 0.6 * level)
        self.n_punches = sround(self.n_punches + 0.2 * level)


def _fold_shape(height, width, direction):
    if direction in {"left", "right"}:
        return height, width // 2
    return height // 2, width


def _unfold(points, folds):
    points = set(points)
    for direction, height, width in reversed(folds):
        expanded = set()
        if direction == "left":
            half = width // 2
            for row, col in points:
                expanded.add((row, half + col))
                expanded.add((row, half - 1 - col))
        elif direction == "right":
            for row, col in points:
                expanded.add((row, col))
                expanded.add((row, width - 1 - col))
        elif direction == "top":
            half = height // 2
            for row, col in points:
                expanded.add((half + row, col))
                expanded.add((half - 1 - row, col))
        else:
            for row, col in points:
                expanded.add((row, col))
                expanded.add((height - 1 - row, col))
        points = expanded
    return points


def _format_points(points):
    return "; ".join(f"{row + 1},{col + 1}" for row, col in sorted(points))


class SpatialFolding(GeneratedMixin, Task):
    summary = "Track hole positions while folding and unfolding a square grid."
    config_cls = SpatialFoldingConfig

    def generate_entry(self):
        n_folds = min(4, max(1, int(self.config.n_folds)))
        n_punches = min(2, max(1, int(self.config.n_punches)))
        side = 2 ** max(2, n_folds)
        height = width = side
        folds = []
        directions = []

        for _ in range(n_folds):
            candidates = []
            if width % 2 == 0 and width > 1:
                candidates += ["left", "right"]
            if height % 2 == 0 and height > 1:
                candidates += ["top", "bottom"]
            direction = random.choice(candidates)
            folds.append((direction, height, width))
            directions.append(direction)
            height, width = _fold_shape(height, width, direction)

        positions = [(r, c) for r in range(height) for c in range(width)]
        punches = random.sample(positions, min(n_punches, len(positions)))
        holes = _unfold(punches, folds)
        metadata = edict(
            side=side,
            folds=directions,
            folded_shape=(height, width),
            punches=punches,
            holes=sorted(holes),
        )
        return Entry(metadata=metadata, answer=_format_points(holes))

    def render_prompt(self, m):
        fold_text = "\n".join(f"{i}. {FOLD_TEXT[d]}." for i, d in enumerate(m.folds, 1))
        punches = "; ".join(f"{r + 1},{c + 1}" for r, c in m.punches)
        return (
            f"A {m.side}x{m.side} square sheet is divided into unit cells. Rows are numbered top-to-bottom "
            "and columns left-to-right, starting at 1. After every fold, renumber the visible folded rectangle "
            "from its new top-left corner.\n"
            f"Folds, in order:\n{fold_text}\n"
            f"After all folds the sheet is {m.folded_shape[0]}x{m.folded_shape[1]}. Punch holes through cells: {punches}.\n"
            "Unfold the sheet completely. The answer is all punched cells as row,column pairs separated by semicolons, "
            "in row-major order."
        )

    def score_answer(self, answer, entry):
        pairs = re.findall(r"(\d+)\s*,\s*(\d+)", str(answer))
        parsed = [(int(r), int(c)) for r, c in pairs]
        gold = [(r + 1, c + 1) for r, c in entry.metadata.holes]
        return float(len(parsed) == len(gold) and set(parsed) == set(gold))

    def balancing_key(self, problem):
        return len(problem.metadata.holes)
