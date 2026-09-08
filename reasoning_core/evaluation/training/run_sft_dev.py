#!/usr/bin/env python
"""Small parallel SFT planner for testing the experimental shared arm runner."""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from reasoning_core.evaluation.training.paths import configure_runtime_env

configure_runtime_env()

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning_core.evaluation.training.data import (
    FORMATTERS, StreamSpec, format_row, load_stream, mix_streams, ratio_to_fraction,
    settle_remote_streams, source_id, steps_for_token_budget,
)
from reasoning_core.evaluation.training.arm import ArmSpec, record_event, run_arm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sileod/microlm-ettin-swa-5m")
    parser.add_argument("--optimizer", choices=("prodigy", "adamc"), default="prodigy")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--checkpoint-every-minutes", type=float, default=60)
    parser.add_argument("--experiment-id", default="run-sft-dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initialization-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--main-source")
    parser.add_argument("--main-data-id")
    parser.add_argument("--main-revision")
    parser.add_argument("--main-config")
    parser.add_argument("--main-format", choices=FORMATTERS, default="sft_qa_v1")
    parser.add_argument("--aux-source")
    parser.add_argument("--aux-data-id")
    parser.add_argument("--aux-revision")
    parser.add_argument("--aux-config")
    parser.add_argument("--aux-format", choices=FORMATTERS, default="sft_qa_v1")
    parser.add_argument("--main-prefix", default="")
    parser.add_argument("--aux-prefix", default="")
    parser.add_argument("--aux-ratio", type=float, default=0.2)
    parser.add_argument("--eval-examples", type=int, default=16)
    args = parser.parse_args()

    def identify(source, supplied=None, revision=None, synthetic=None):
        if not source:
            return synthetic
        try:
            return source_id(source, supplied, revision)
        except ValueError as error:
            parser.error(str(error))

    initialization_id = identify(args.model, args.initialization_id, args.model_revision)
    main_data_id = identify(
        args.main_source, args.main_data_id, args.main_revision,
        synthetic="synthetic:arithmetic-v1",
    )
    aux_data_id = identify(args.aux_source, args.aux_data_id, args.aux_revision) or ""

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = [format_row(
        {"prompt": q, "answer": a}, tokenizer.eos_token, args.main_format, args.main_prefix,
    ) for q, a in (
        ("1 + 1?", "2"), ("2 + 2?", "4"), ("3 + 3?", "6"), ("4 + 4?", "8"),
    )]
    if args.main_source:
        main = load_stream(StreamSpec(
            args.main_source, args.main_format, config=args.main_config,
            revision=args.main_revision, prompt_prefix=args.main_prefix,
        ),
                           tokenizer, max_length=args.max_length)
        eval_dataset = Dataset.from_list(list(main.take(args.eval_examples)))
        aux = (load_stream(StreamSpec(
            args.aux_source, args.aux_format, config=args.aux_config, cycle=True,
            revision=args.aux_revision, prompt_prefix=args.aux_prefix,
        ), tokenizer, max_length=args.max_length)
               if args.aux_source else None)
        dataset = mix_streams(
            main, aux, ratio_to_fraction(args.aux_ratio), seed=args.seed, shuffle_buffer=100,
        )
    else:
        dataset = eval_dataset = Dataset.from_list(rows)
    steps = args.steps
    if args.token_budget:
        steps = steps_for_token_budget(
            args.token_budget, args.aux_ratio if args.aux_source else 0,
            args.max_length, args.batch_size * args.gradient_accumulation_steps,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, dtype=torch.float32,
    ).to("cuda")
    spec = ArmSpec(
        experiment_id=args.experiment_id, arm_id="stage1", model=args.model,
        model_revision=args.model_revision,
        optimizer=args.optimizer,
        max_steps=steps, batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length, checkpoint_every_minutes=args.checkpoint_every_minutes,
        save_final=True, formatter=args.main_format,
        aux_formatter=args.aux_format if args.aux_source else None,
        prompt_prefix=args.main_prefix, aux_prompt_prefix=args.aux_prefix,
        main_source=args.main_source or "synthetic", main_config=args.main_config,
        main_revision=args.main_revision,
        aux_source=args.aux_source, aux_config=args.aux_config, aux_revision=args.aux_revision,
        aux_fraction=ratio_to_fraction(args.aux_ratio) if args.aux_source else 0,
        shuffle_buffer=100, seed=args.seed, initialization_id=initialization_id,
        main_data_id=main_data_id, aux_data_id=aux_data_id,
        eval_ids=("dev/main_nll@v1",),
    )
    _, metrics = run_arm(model, tokenizer, dataset, spec, eval_dataset=eval_dataset)
    if metrics:
        record_event(spec, "stage_complete", metrics)
        print(metrics)
    if args.main_source and not Path(args.main_source).expanduser().exists():
        settle_remote_streams()


if __name__ == "__main__":
    main()
