from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager, get_task_dict
from tabulate import tabulate

from reasoning_core.downstream_eval import logic_custom_task_configs, pick_metric


BUILTIN_LOGIC_TASKS = (
    "anli_r1",
    "anli_r2",
    "anli_r3",
    "mnli",
    "rte",
    "qnli",
    "cb",
)

CUSTOM_LOGIC_TASKS = (
    "wanli", "hans", "nan_nli", "folio", "logiqa2_nli",
    "semantic_fragments_nli", "control_nli", "commonsense_qa_2",
    "math_qa", "gsm8k_mc", "infotabs", "reclor", "boardgameqa",
)
LOGIC_TASKS = BUILTIN_LOGIC_TASKS + CUSTOM_LOGIC_TASKS


CUSTOM_TASK_CONFIGS = {
    **logic_custom_task_configs,
    "reclor": {
        "task": "reclor",
        "dataset_path": "reclor",
        "validation_split": "validation",
        "output_type": "multiple_choice",
        "doc_to_text": "{{context}}\nQuestion: {{question}}\nAnswer:",
        "doc_to_choice": "{{answers}}",
        "doc_to_target": '{{["A", "B", "C", "D"].index(label)}}',
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
    },
}


@dataclass
class LogicResult:
    task: str
    metric: str
    score: float | None
    n: int
    seconds: float
    status: str = "ok"


def _metric_name(metrics: dict) -> str:
    return next((k for k in ("mcc,none", "acc_norm,none", "acc,none") if k in metrics), "")


def _print_table(rows: list[LogicResult]) -> None:
    print(
        tabulate(
            [
                [
                    r.task,
                    r.metric,
                    "" if r.score is None else f"{r.score:.4f}",
                    r.n,
                    f"{r.seconds:.1f}",
                    r.status,
                ]
                for r in rows
            ],
            headers=["task", "metric", "score", "limit", "sec", "status"],
        )
    )


def available_logic_tasks(tasks: tuple[str, ...] = LOGIC_TASKS) -> list[str]:
    manager = TaskManager()
    return [task for task in tasks if manager.match_tasks([task]) or task in CUSTOM_TASK_CONFIGS]


def evaluate_logic(
    model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    tasks: tuple[str, ...] = LOGIC_TASKS,
    limit: int = 5,
    batch_size: int | str = 1,
    device: str = "cpu",
    timeout_s: float | None = None,
) -> dict[str, float]:
    hflm = HFLM(pretrained=model, batch_size=batch_size, device=device)
    manager = TaskManager()
    rows: list[LogicResult] = []

    for task in tasks:
        start = time.time()
        try:
            if task in CUSTOM_TASK_CONFIGS:
                task_dict = {task: ConfigurableTask(config=CUSTOM_TASK_CONFIGS[task])}
            elif manager.match_tasks([task]):
                task_dict = get_task_dict([task], manager)
            else:
                raise ValueError("not found in lm-eval")
            metrics = evaluate(lm=hflm, task_dict=task_dict, limit=limit)["results"][task]
            metric = _metric_name(metrics)
            rows.append(LogicResult(task, metric, pick_metric(metrics), limit, time.time() - start))
        except Exception as exc:
            rows.append(LogicResult(task, "", None, limit, time.time() - start, f"skip: {exc}"))

        _print_table(rows)
        if timeout_s is not None and rows[-1].seconds > timeout_s:
            print(f"aborting after {task}: exceeded {timeout_s}s")
            break

    return {r.task: r.score for r in rows if r.score is not None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--tasks", nargs="*", default=list(LOGIC_TASKS))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout-s", type=float, default=None)
    args = parser.parse_args()

    batch_size: int | str = int(args.batch_size) if args.batch_size.isdigit() else args.batch_size
    evaluate_logic(
        model=args.model,
        tasks=tuple(args.tasks),
        limit=args.limit,
        batch_size=batch_size,
        device=args.device,
        timeout_s=args.timeout_s,
    )


if __name__ == "__main__":
    main()
