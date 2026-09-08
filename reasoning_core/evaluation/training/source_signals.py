import torch
import torch.nn.functional as F
from datasets import IterableDataset
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


def add_source(ds, source):
    if ds is None:
        return None
    return ds.map(lambda ex: {**ex, "train_source": source})


def pack_by_source(ds, tokenizer, max_length):
    def gen():
        buffers = {0: {"input_ids": [], "completion_mask": []}, 1: {"input_ids": [], "completion_mask": []}}
        for ex in ds:
            src = int(ex.get("train_source", 0))
            prompt_ids = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
            input_ids = tokenizer(ex["prompt"] + ex["completion"], add_special_tokens=False)["input_ids"]
            if len(input_ids) > max_length:
                continue
            mask = [0] * min(len(prompt_ids), len(input_ids)) + [1] * max(0, len(input_ids) - len(prompt_ids))
            buf = buffers[src]
            buf["input_ids"].extend(input_ids)
            buf["completion_mask"].extend(mask)
            while len(buf["input_ids"]) >= max_length:
                yield {
                    "input_ids": buf["input_ids"][:max_length],
                    "completion_mask": buf["completion_mask"][:max_length],
                    "source_ids": [src] * max_length,
                }
                del buf["input_ids"][:max_length]
                del buf["completion_mask"][:max_length]

    return IterableDataset.from_generator(gen)


class SourceDataCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        source_ids = [ex.pop("source_ids") for ex in examples] if "source_ids" in examples[0] else None
        batch = super().torch_call(examples)
        if source_ids is None:
            return batch
        ids = [torch.tensor(x[: batch["input_ids"].shape[1]]) for x in source_ids]
        batch["source_ids"] = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=-1)
        return batch


class SourceLossMixin:
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        source_ids = inputs.pop("source_ids", None)
        labels = inputs.get("labels")
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        should_sample = (self.state.global_step + 1) % max(1, self.args.logging_steps) == 0
        if should_sample and self.model.training and source_ids is not None and labels is not None:
            with torch.no_grad():
                losses = F.cross_entropy(
                    outputs.logits[..., :-1, :].contiguous().view(-1, outputs.logits.shape[-1]),
                    labels[..., 1:].contiguous().view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(labels[..., 1:])
                shifted_sources = source_ids[..., 1:].to(losses.device)
                mask = labels[..., 1:].ne(-100).to(losses.device)
                for src, name in ((0, "train_source/main_loss"), (1, "train_source/aux_loss")):
                    src_mask = mask & shifted_sources.eq(src)
                    if src_mask.any():
                        self._metrics["train"][name].append(losses[src_mask].mean().item())
        return (loss, outputs) if return_outputs else loss
