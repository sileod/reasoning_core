"""Test a weighted-linear prediction of mixture scores against actual measurements."""

from dataclasses import dataclass
import math

from .groups import TaskGroup


@dataclass(frozen=True)
class GroupMeasurement:
    group: TaskGroup
    protocol_id: str
    seed: int
    metrics: dict[str, float]

    def __post_init__(self):
        if not self.protocol_id or not self.metrics:
            raise ValueError("Measurements require a protocol identity and metrics")
        if not all(math.isfinite(v) for v in self.metrics.values()):
            raise ValueError("Measurement metrics must be finite")


def compare_composition(observed, components):
    """Predict a group's scores from a disjoint partition of constituent measurements.

    Inputs must share the evaluation/training protocol and seed. The protocol ID
    must identify dose, initialization, task-data snapshots, battery, formatting,
    and metric definitions.
    Residual is observed minus predicted in the original metric's units; the linear
    prediction is a hypothesis, not an assertion that training effects are additive.
    """
    target = dict(zip(observed.group.tasks, observed.group.weights))
    used = set()
    predicted = dict.fromkeys(observed.metrics, 0.0)
    for component in components:
        if (component.protocol_id != observed.protocol_id or component.seed != observed.seed
                or component.metrics.keys() != observed.metrics.keys()):
            raise ValueError("Composition requires matching protocols, seeds, and metrics")
        members = set(component.group.tasks)
        if not members <= target.keys() or members & used:
            raise ValueError("Components must form a disjoint partition of the measured group")
        mass = sum(target[t] for t in members)
        for task, weight in zip(component.group.tasks, component.group.weights):
            if not math.isclose(weight, target[task] / mass):
                raise ValueError("Component weights must match their share of the measured group")
        for metric, value in component.metrics.items():
            predicted[metric] += mass * value
        used.update(members)
    if used != target.keys():
        raise ValueError("A constituent measurement is missing")
    return {"predicted": predicted, "observed": dict(observed.metrics),
            "residual": {m: observed.metrics[m] - predicted[m] for m in predicted}}
