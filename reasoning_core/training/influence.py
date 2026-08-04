"""Paired influence experiments built from the canonical arm runner."""

from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Real

from reasoning_core.training.arm import ArmSpec, run_arm


@dataclass(frozen=True)
class ArmPlan:
    spec: ArmSpec
    dataset: Callable[[], object]
    evaluate_endpoint: Callable[[object], dict] | None = None


@dataclass(frozen=True)
class InfluenceResult:
    baseline: dict
    treatments: dict[str, dict]
    metric_names: tuple[str, ...]
    initial: dict = field(default_factory=dict)

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
                  evaluate=None, evaluate_endpoints=None, evaluate_initial=None):
    """Run one baseline and any number of treatments from identical weights."""

    plans = (baseline, *treatments)
    arm_ids = [plan.spec.arm_id for plan in plans]
    if len(arm_ids) != len(set(arm_ids)):
        raise ValueError("Influence arm IDs must be unique")
    metric_names = tuple(metric_names)
    if not metric_names:
        raise ValueError("metric_names must name at least one scientific outcome")
    initial_state = {
        name: value.detach().cpu().clone() if hasattr(value, "detach") else value
        for name, value in initial_state.items()
    }
    initial = {}
    initial_evaluator = evaluate_initial or evaluate_endpoints
    if initial_evaluator is not None:
        model.load_state_dict(initial_state)
        initial = initial_evaluator(model)

    results = {}
    for plan in plans:
        endpoint = plan.evaluate_endpoint or evaluate_endpoints

        def evaluate_final(current_model, endpoint=endpoint):
            metrics = evaluate(current_model) if evaluate is not None else {}
            endpoint_metrics = endpoint(current_model) if endpoint is not None else {}
            overlap = metrics.keys() & endpoint_metrics.keys()
            if overlap:
                raise ValueError(f"Duplicate evaluation metrics: {', '.join(sorted(overlap))}")
            return {**metrics, **endpoint_metrics}

        arm_evaluate = evaluate_final if evaluate is not None or endpoint is not None else None
        model.load_state_dict(initial_state)
        _, metrics = run_arm(
            model, tokenizer, plan.dataset(), plan.spec,
            evaluate=arm_evaluate,
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
    return InfluenceResult(baseline_metrics, results, metric_names, initial)
