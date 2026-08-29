import asyncio
import os

from datasets import Dataset, DatasetDict
from easydict import EasyDict as edict
from verifiers.types import ClientConfig

from reasoning_core import get_task, score_answer
from reasoning_core.primeintellect.reasoning_core_env.reasoning_core_env import rc_ds_to_env


def client_config():
    os.environ.setdefault("DUMMY_API_KEY", "dummy")
    return ClientConfig(
        client_type="openai_chat_completions",
        api_key_var="DUMMY_API_KEY",
        api_base_url="http://localhost",
    )


example = get_task("arithmetics").generate_example()
row = example.to_dict()
env = rc_ds_to_env(DatasetDict({
    "train": Dataset.from_list([row]),
    "test": Dataset.from_list([row]),
}))

assert len(env.dataset) == 1
assert len(env.eval_dataset) == 1

item = env.dataset[0]
assert score_answer(item["answer"], edict(item["info"])) == 1
assert score_answer("__wrong__", edict(item["info"])) < 1

state = asyncio.run(env.init_state(item, client_config(), "dummy-model"))
assert state["answer"] == item["answer"]
print("prime env smoke ok")
