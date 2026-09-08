import json
import pandas as pd
from datasets import Dataset, load_dataset, DatasetDict, get_dataset_split_names
import verifiers as vf
import reasoning_gym as rg
from reasoning_core import list_tasks, score_answer
from easydict import EasyDict as edict

DEFAULT_SYSTEM_PROMPT = rg.utils.SYSTEM_PROMPTS["DeepSeekZero"]


def _metadata_dict(example):
    metadata = example.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    return metadata or {}


def _example_task_name(example):
    metadata = _metadata_dict(example)
    return (
        metadata.get("_task")
        or example.get("task")
        or metadata.get("task")
    )


def _filter_available_tasks(dataset, available_tasks=None):
    if available_tasks is None:
        available_tasks = list_tasks()
    available_tasks = set(available_tasks)
    before = len(dataset)
    dataset = dataset.filter(lambda example: _example_task_name(example) in available_tasks)
    dropped = before - len(dataset)
    if dropped:
        print(f"Ignored {dropped} examples with unavailable tasks.")
    return dataset


def _prepare_env_dataset(dataset, available_tasks=None):
    dataset = _filter_available_tasks(
        dataset,
        available_tasks=available_tasks,
    ).rename_columns({"prompt": "question"})

    def parse_entry(example):
        entry = edict(
            metadata=example.get('metadata', {}),
            answer=example['answer']
        )
        return {'info': entry}

    dataset = dataset.map(parse_entry)
    return dataset.select_columns(
        [column for column in ("question", "answer", "info") if column in dataset.column_names]
    )


def extract_answer(s, tag="answer"):
    if s.strip().startswith('<prompt>'):
        s = s.split('</prompt>')[-1]
    return s.split(f'<{tag}>')[-1].split(f'</{tag}>')[0]


def rc_ds_to_env(
    ds, 
    system_prompt=DEFAULT_SYSTEM_PROMPT, 
    do_extract_answer=True
):
    """
    Convert Dataset or DatasetDict into a verifiers SingleTurnEnv.
    Properly handles DatasetDict by separating train and eval splits.
    """
    if isinstance(ds, DatasetDict):
        # Main dataset (used for training/sampling)
        if "train" in ds:
            main_ds = ds["train"]
        else:
            main_ds = ds[list(ds.keys())[0]]  # fallback

        # Evaluation dataset
        eval_ds = None
        for split_name in ["test", "validation", "eval", "dev"]:
            if split_name in ds:
                eval_ds = ds[split_name]
                break
    else:
        # If a single Dataset is passed
        main_ds = ds
        eval_ds = None

    # Process main dataset
    dataset = _prepare_env_dataset(main_ds)

    # Process eval dataset if it exists
    eval_dataset = None
    if eval_ds is not None:
        eval_dataset = _prepare_env_dataset(eval_ds)

    def score_answer_vf(prompt, completion, info) -> float:
        answer = completion[0]['content']
        if do_extract_answer:
            answer = extract_answer(answer)
        return score_answer(answer, edict(info))

    rubric = vf.Rubric(funcs=[score_answer_vf])

    env = vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,      # ← Important: use eval_dataset
        rubric=rubric,
        system_prompt=system_prompt
    )
    return env


def load_environment(
    num_train_examples: int = 500,
    num_eval_examples: int = 50,
    do_extract_answer=True,
    dataset_name: str = "reasoning-core/formal-reasoning-env",
    seed: int = 0,
) -> vf.SingleTurnEnv:
    """
    Load the dataset and return a ready-to-use SingleTurnEnv.
    """
    available_splits = get_dataset_split_names(dataset_name)
    eval_source_split = next(
        (split for split in ("test", "validation", "eval", "dev") if split in available_splits),
        "train",
    )

    splits = {
        "train": ("train", num_train_examples, seed),
        "test": (eval_source_split, num_eval_examples, seed + int(eval_source_split == "train")),
    }

    ds = DatasetDict({
        split: Dataset.from_list(
            list(
                load_dataset(dataset_name, split=source_split, streaming=True)
                .shuffle(seed=split_seed)
                .take(n)
            )
        )
        for split, (source_split, n, split_seed) in splits.items()
    })

    return rc_ds_to_env(ds, do_extract_answer=do_extract_answer)


# ============================
# Usage Example
# ============================

if __name__ == "__main__":
    env = load_environment(
        num_train_examples=500,
        num_eval_examples=50,
        seed=42
    )
    
    print(f"Environment created with {len(env.dataset)} training examples")
    if env.eval_dataset is not None:
        print(f"and {len(env.eval_dataset)} eval examples")
