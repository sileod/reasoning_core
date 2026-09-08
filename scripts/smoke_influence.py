"""Offline CPU-sized evaluation and paired training example (not a benchmark)."""

import argparse
import hashlib
import json
from dataclasses import replace


def content_hash(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from reasoning_core.evaluation.training.arm import ArmSpec
    from reasoning_core.evaluation.metrics import EvalExample, EvalSuite, evaluate_qa_nll
    from reasoning_core.evaluation.influence import ArmPlan, run_influence
    from reasoning_core.evaluation.groups import TaskGroup
    from reasoning_core.evaluation.training.groups import group_arm

    torch.set_num_threads(1)
    torch.manual_seed(0)
    words = ["[UNK]", "[PAD]", "[EOS]", "value", "0", "1", "2", "3"]
    backend = Tokenizer(models.WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                       pad_token="[PAD]", eos_token="[EOS]")
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(words), n_positions=32, n_embd=16, n_layer=1, n_head=1,
        bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ))
    suite = EvalSuite("smoke", (EvalExample("value 2 ", "2 [EOS]"),))

    def evaluate(current):
        return {"nll": evaluate_qa_nll(current, tokenizer, suite.examples, 32)["nll"]}

    if args.eval_only:
        print(json.dumps({"eval_id": suite.identifier, **evaluate(model)}))
        return

    main_rows = [{"prompt": f"value {n} ", "completion": f"{n} [EOS]"} for n in (0, 1)] * 4
    initial = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
    digest = hashlib.sha256(json.dumps(model.config.to_dict(), sort_keys=True).encode())
    for key, tensor in sorted(initial.items()):
        digest.update(key.encode())
        digest.update(tensor.numpy().tobytes())
    baseline_spec = ArmSpec(
        "workflow-smoke-v1", "baseline", optimizer="adamw_torch", learning_rate=0.001,
        max_steps=2, batch_size=2, max_length=32, packing=False,
        initialization_id="sha256:" + digest.hexdigest(), main_data_id=content_hash(main_rows),
        eval_ids=(suite.identifier,), formatter="smoke_text_v1",
    )
    group = TaskGroup(("toy_2", "toy_3"))
    task_rows = {f"toy_{n}": [{"prompt": f"value {n}", "answer": f"{n} "}] for n in (2, 3)}
    treatment = group_arm(
        replace(baseline_spec, arm_id="treatment", aux_formatter="influence_legacy_v1"),
        group, main_rows, task_rows, tokenizer, aux_token_fraction=0.5,
    )
    treatment_spec = treatment.spec
    result = run_influence(
        model, tokenizer, initial,
        ArmPlan(baseline_spec, lambda: Dataset.from_list(main_rows)),
        (treatment,),
        metric_names=("nll",), evaluate=evaluate,
    )
    print(json.dumps({"baseline": result.baseline, "treatments": result.treatments,
                      "deltas": result.deltas, "run_dirs": [str(baseline_spec.run_dir),
                                                            str(treatment_spec.run_dir)]}, indent=2))


if __name__ == "__main__":
    main()
