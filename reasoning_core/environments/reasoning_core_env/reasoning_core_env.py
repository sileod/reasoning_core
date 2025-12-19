import json
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets
import verifiers as vf
import reasoning_gym as rg
from reasoning_core import generate_dataset, score_answer
from easydict import EasyDict as edict

def load_environment(
    num_train_examples: int = 500,
    num_eval_examples: int = 50,
    rebuild: bool = False,
    dataset_name: str = "reasoning-core/rc1",
    seed: int = 0,
):
    if rebuild:
        total = num_train_examples + num_eval_examples
        D = generate_dataset(total)
        df = pd.DataFrame(D)
        df["metadata"] = df["metadata"].map(json.dumps)
        ds = Dataset.from_pandas(df).train_test_split(test_size=num_eval_examples)
    else:
        ds = load_dataset(
            dataset_name,
            split={
                "train": f"train[:{num_train_examples}]",
                "test": f"test[:{num_eval_examples}]",
            },
        )
        if isinstance(ds, Dataset) or "test" not in ds:
            ds = ds.train_test_split(test_size=num_eval_examples)

    for split, v in ds.items():
        ds[split] = ds[split].add_column("split", [split] * len(ds[split]))

    ds["train"] = ds["train"].select(range(num_train_examples))
    ds["test"] = ds["test"].select(range(num_eval_examples))
    ds = concatenate_datasets(list(ds.values()))
    df = ds.to_pandas()

    df["question"] = df["prompt"]
    df["metadata"] = df["metadata"].map(json.dumps)
    del df["prompt"]
    df["index"] = [str(i) for i in range(len(df))]
    df["answer"] = df.apply(lambda r: json.dumps({"index": r["index"], "answer": r["answer"]}), axis=1)

    rc_dataset = Dataset.from_pandas(df)
    dataset = rc_dataset.filter(lambda x: x["split"] == "train")
    eval_dataset = rc_dataset.filter(lambda x: x["split"] == "test")

    parser = vf.XMLParser(fields=["think", "answer"])
    rubric = vf.Rubric(parser=parser)

    def check_answer_reward_func(completion, answer, **kwargs):
        d = json.loads(answer)
        idx = int(d["index"])
        ref = d["answer"]
        entry = rc_dataset[idx]
        entry["metadata"] = json.loads(json.loads(entry["metadata"]))
        entry["answer"] = ref
        resp = str(parser.parse_answer(completion)).strip()
        return score_answer(resp, edict(entry))

    rubric.add_reward_func(check_answer_reward_func)
    rubric.add_reward_func(parser.get_format_reward_func(), weight=0.2)

    system_prompt = rg.utils.SYSTEM_PROMPTS["DeepSeekZero"]

    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        seed=seed,
    )
