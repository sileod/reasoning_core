"""Build a reproducible group treatment using the existing arm runner."""

from dataclasses import replace
from copy import deepcopy
import hashlib
import json
import random

from datasets import IterableDataset

from ..groups import TaskGroup
from ..influence import ArmPlan
from ..intrinsic import group_reward, group_reward_id
from .data import format_row


def group_arm(spec, group, main_rows, task_rows, tokenizer, *, aux_token_fraction, reward_rows=None,
              reward_spec=None, evaluation_group=None):
    """Construct an ArmPlan from formatted main rows and raw per-task auxiliary rows.

    aux_token_fraction is required. Sampling probabilities correct for each
    source's mean token length, giving the requested expected token shares. Realized
    shares vary in short runs. The replayable stream samples with replacement and
    is bounded by ArmSpec.max_steps, not by exhaustion of one component.
    """
    group.require_members(task_rows)
    fraction = aux_token_fraction
    if fraction is None or not 0 < fraction < 1:
        raise ValueError("Set aux_token_fraction between zero and one")
    if spec.shuffle_buffer:
        raise ValueError("group_arm uses seeded sampling; shuffle_buffer must be zero")
    main = [{"prompt": r["prompt"], "completion": r["completion"]} for r in main_rows]
    aux = [[format_row(dict(r), tokenizer.eos_token, spec.aux_formatter or spec.formatter,
                       spec.aux_prompt_prefix) for r in task_rows[t]] for t in group.tasks]
    sources = [main, *aux]
    if any(not rows for rows in sources):
        raise ValueError("Every training source must have rows")
    probabilities = []
    for rows, share in zip(sources, [1 - fraction, *(fraction * w for w in group.weights)]):
        lengths = [len(tokenizer(r["prompt"] + r["completion"])["input_ids"]) for r in rows]
        if min(lengths) < 1 or max(lengths) > spec.max_length:
            raise ValueError("Training rows must fit max_length without truncation")
        probabilities.append(share / (sum(lengths) / len(lengths)))

    def content_id(value):
        return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    endpoint = None
    eval_ids = spec.eval_ids
    if reward_rows is not None:
        if reward_spec is None:
            raise ValueError("reward_rows requires reward_spec")
        evaluation_group = evaluation_group or TaskGroup(group.tasks)
        evaluation_group.require_members(reward_rows)
        reward_rows = deepcopy({t: [dict(r) for r in reward_rows[t]] for t in evaluation_group.tasks})
        eval_ids = (*eval_ids, group_reward_id(reward_rows, evaluation_group, reward_spec, spec.max_length))
        endpoint = lambda model: group_reward(model, tokenizer, reward_rows, evaluation_group,
                                              reward_spec, spec.max_length)
    elif reward_spec is not None or evaluation_group is not None:
        raise ValueError("Intrinsic evaluation requires fixed reward_rows")
    actual = replace(spec, aux_tasks=group.tasks, aux_weights=group.weights,
                     target_aux_token_fraction=fraction,
                     aux_source=group.identifier, aux_task=None,
                     main_data_id=content_id(main),
                     aux_data_id=content_id(["group_sampling_v1", group.identifier, aux, probabilities]),
                     aux_fraction=sum(probabilities[1:]) / sum(probabilities), eval_ids=eval_ids)

    def stream():
        rng = random.Random(actual.seed)
        while True:
            source = rng.choices(sources, weights=probabilities, k=1)[0]
            yield dict(rng.choice(source))

    return ArmPlan(actual, lambda: IterableDataset.from_generator(stream), evaluate_endpoint=endpoint,
                   evaluate_initial=endpoint)
