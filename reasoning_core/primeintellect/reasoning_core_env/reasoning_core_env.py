import json
import pandas as pd
from datasets import Dataset, load_dataset, DatasetDict
import verifiers as vf
import reasoning_gym as rg
from reasoning_core import score_answer
from easydict import EasyDict as edict

DEFAULT_SYSTEM_PROMPT = rg.utils.SYSTEM_PROMPTS["DeepSeekZero"]


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
    dataset = main_ds.rename_columns({"prompt": "question"})

    def parse_entry(example):
        entry = edict(
            metadata=example.get('metadata', {}),
            answer=example['answer']
        )
        return {'info': entry}

    dataset = dataset.map(parse_entry)

    # Process eval dataset if it exists
    eval_dataset = None
    if eval_ds is not None:
        eval_dataset = eval_ds.rename_columns({"prompt": "question"})
        eval_dataset = eval_dataset.map(parse_entry)

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
    dataset_name: str = "reasoning-core/rc1",
    seed: int = 0,
) -> vf.SingleTurnEnv:
    """
    Load the dataset and return a ready-to-use SingleTurnEnv.
    """
    splits = {"train": num_train_examples, "test": num_eval_examples}

    ds = DatasetDict({
        split: Dataset.from_list(
            list(
                load_dataset(dataset_name, split=split, streaming=True)
                .shuffle(seed=seed)
                .take(n)
            )
        )
        for split, n in splits.items()
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
