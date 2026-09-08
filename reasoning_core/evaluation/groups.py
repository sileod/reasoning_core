"""Named, weighted task groups shared by training, rewards, and composition analysis."""

from dataclasses import dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class TaskGroup:
    tasks: tuple[str, ...]
    weights: tuple[float, ...] = ()

    def __post_init__(self):
        if isinstance(self.tasks, str):
            raise ValueError("Pass a sequence of task names, not a single string")
        tasks = tuple(self.tasks)
        weights = tuple(self.weights) or (1.0,) * len(tasks)
        if (not tasks or any(not isinstance(t, str) or not t.strip() for t in tasks)
                or len(set(tasks)) != len(tasks) or len(tasks) != len(weights)):
            raise ValueError("A group needs distinct task names and one weight per task")
        if any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("Group weights must be finite and positive")
        scale = max(weights)
        total = math.fsum(w / scale for w in weights)
        pairs = sorted(zip(tasks, (w / scale / total for w in weights)))
        object.__setattr__(self, "tasks", tuple(t for t, _ in pairs))
        object.__setattr__(self, "weights", tuple(w for _, w in pairs))

    @property
    def identifier(self):
        data = json.dumps(list(zip(self.tasks, self.weights)), separators=(",", ":"))
        return "group@v1:" + hashlib.sha256(data.encode()).hexdigest()[:16]

    def require_members(self, values):
        if set(values) != set(self.tasks):
            raise ValueError(f"Expected exactly these group members: {self.tasks}")
