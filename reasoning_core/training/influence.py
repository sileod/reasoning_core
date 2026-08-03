"""Paired influence experiments built from the canonical arm runner."""

from collections.abc import Callable
from dataclasses import dataclass

from reasoning_core.training.arm import ArmSpec, run_arm


@dataclass(frozen=True)
class ArmPlan:
    spec: ArmSpec
    dataset: Callable[[], object]


@dataclass(frozen=True)
class InfluenceResult:
    baseline: dict
    treatments: dict[str, dict]

    @property
    def deltas(self):
        return {
            arm: {
                metric: value - self.baseline[metric]
                for metric, value in metrics.items()
                if isinstance(value, (int, float))
                and isinstance(self.baseline.get(metric), (int, float))
            }
            for arm, metrics in self.treatments.items()
        }


def run_influence(model, tokenizer, initial_state, baseline, treatments, *, evaluate=None):
    """Run one baseline and any number of treatments from identical weights."""

    results = {}
    for plan in (baseline, *treatments):
        model.load_state_dict(initial_state)
        _, metrics = run_arm(
            model, tokenizer, plan.dataset(), plan.spec, evaluate=evaluate,
        )
        if metrics is None:
            raise RuntimeError(f"Arm {plan.spec.arm_id!r} was interrupted")
        results[plan.spec.arm_id] = metrics
    baseline_metrics = results.pop(baseline.spec.arm_id)
    return InfluenceResult(baseline_metrics, results)
