"""Paired influence experiments built from the canonical arm runner."""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real

from reasoning_core.training.arm import ArmSpec, run_arm


@dataclass(frozen=True)
class ArmPlan:
    spec: ArmSpec
    dataset: Callable[[], object]


@dataclass(frozen=True)
class InfluenceResult:
    baseline: dict
    treatments: dict[str, dict]
    metric_names: tuple[str, ...]

    @property
    def deltas(self):
        return {
            arm: {
                metric: metrics[metric] - self.baseline[metric]
                for metric in self.metric_names
            }
            for arm, metrics in self.treatments.items()
        }


def run_influence(model, tokenizer, initial_state, baseline, treatments, *, metric_names,
                  evaluate=None):
    """Run one baseline and any number of treatments from identical weights."""

    plans = (baseline, *treatments)
    arm_ids = [plan.spec.arm_id for plan in plans]
    if len(arm_ids) != len(set(arm_ids)):
        raise ValueError("Influence arm IDs must be unique")
    metric_names = tuple(metric_names)
    if not metric_names:
        raise ValueError("metric_names must name at least one scientific outcome")
    results = {}
    for plan in plans:
        model.load_state_dict(initial_state)
        _, metrics = run_arm(
            model, tokenizer, plan.dataset(), plan.spec, evaluate=evaluate,
        )
        if metrics is None:
            raise RuntimeError(f"Arm {plan.spec.arm_id!r} was interrupted")
        invalid = [name for name in metric_names
                   if not isinstance(metrics.get(name), Real)
                   or isinstance(metrics.get(name), bool)]
        if invalid:
            raise ValueError(
                f"Arm {plan.spec.arm_id!r} lacks numeric metrics: {', '.join(invalid)}"
            )
        results[plan.spec.arm_id] = metrics
    baseline_metrics = results.pop(baseline.spec.arm_id)
    return InfluenceResult(baseline_metrics, results, metric_names)
