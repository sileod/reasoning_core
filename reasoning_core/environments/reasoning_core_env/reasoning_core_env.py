import json
import pandas as pd
from typing import List
from reasoning_core import generate_dataset, score_answer
from datasets import Dataset
import reasoning_gym as rg

from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.rubric import Rubric
from easydict import EasyDict as edict
from datasets import load_dataset, concatenate_datasets



def build_dataset(total_examples):
    return generate_dataset(total_examples)

class ReasoningCoreEnv(SingleTurnEnv):
    def __init__(
        self,
        rebuild: bool=False,
        dataset_name = "reasoning-core/rc1",
        num_train_examples: int = 1000,
        num_eval_examples: int = 100,
        seed: int = 0,
        **kwargs,
    ):
        self.num_train_examples = num_train_examples
        self.num_eval_examples = num_eval_examples
        self.seed = seed

        if rebuild:
            total_examples = num_train_examples + num_eval_examples
            D = generate_dataset(total_examples)
            df=pd.DataFrame(D)
            df['metadata']=df.metadata.map(json.dumps)
            ds = Dataset.from_pandas(df).train_test_split(test_size=num_eval_examples)

        else:
            ds = load_dataset(
                dataset_name,
                split={
                    "train": f"train[:{num_train_examples}]",
                    "test": f"test[:{num_eval_examples}]"
                })
            if type(ds)==Dataset or "test" not in ds:
                ds=ds.train_test_split(test_size=num_eval_examples)

        for split,v in ds.items():
            ds[split]=ds[split].add_column('split',[split]*len(ds[split]))
        ds['train']=ds['train'].select(range(num_train_examples))
        ds['test']=ds['test'].select(range(num_eval_examples))
        ds=concatenate_datasets(ds.values())
        df=ds.to_pandas()

        df['question'] = df['prompt']
        df['metadata']=df['metadata'].map(json.dumps)
        del df['prompt']

        df['index'] = [str(i) for i in range(len(df))]
        def create_answer_json(row):
            return json.dumps({'index': row['index'], 'answer': row['answer']})

        df['answer'] = df.apply(create_answer_json, axis=1)

        self.rc_dataset = Dataset.from_pandas(df)
        dataset = self.rc_dataset.filter(lambda x:x['split']=='train')
        eval_dataset =  self.rc_dataset.filter(lambda x:x['split']=='test')

        parser = XMLParser(fields=["think", "answer"])
        rubric = Rubric(parser=parser)

        def check_answer_reward_func(completion, answer, **kwargs) -> float:
            answer_data = json.loads(answer)
            index = int(answer_data['index'])
            reference_answer = answer_data['answer']

            entry = self.rc_dataset[index]
            entry['metadata']= json.loads(json.loads(entry['metadata']))
            entry['answer'] = reference_answer
            response = str(parser.parse_answer(completion)).strip()
            reward = score_answer(response, edict(entry))
            return reward

        rubric.add_reward_func(check_answer_reward_func)
        rubric.add_reward_func(parser.get_format_reward_func(), weight=0.2)
        system_prompt = rg.utils.SYSTEM_PROMPTS["DeepSeekZero"]
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            **kwargs,
        )
        self.parser = parser
        self.rubric = rubric

def load_environment(
    num_train_examples: int = 500,
    num_eval_examples: int = 50,
    **kwargs,
):
    vf_env = ReasoningCoreEnv(
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        **kwargs,
    )
    return vf_env
