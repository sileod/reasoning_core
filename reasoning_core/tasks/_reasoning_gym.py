from reasoning_core.template import Task, Problem, Config
from dataclasses import dataclass
import reasoning_gym
import random
import json


@dataclass
class RGConfig(Config):
    rg_task: str = ""
    rg_level: int = 1

    def update(self, c):
        self.rg_level+=c

class Reasoning_Gym(Task):
    def __init__(self, config=RGConfig()):

        datasets=list(reasoning_gym.factory.DATASETS.keys())
        datasets.remove('composite')
        self.datasets = datasets
        super().__init__(config)
        
    def generate(self):
        meta = dict()
        if self.config.rg_task:
            d = self.config.rg_task
        else:
            d = random.choice(self.datasets)

        t,c = reasoning_gym.factory.DATASETS[d]
        c=c()
        if d in reasoning_gym.factory.CURRICULA:
            cl = reasoning_gym.factory.CURRICULA[d]()
            a=random.choice(list(cl.attributes.keys()))
            for k in range(int(self.config.rg_level)):
                cl.increment_attr_level(a)
            c2=cl.generate_configuration()
        else:
            self.config.level=0
        entry =t(c)[0]

        meta = entry['metadata'] | dict(task_name=f"RG.{d}") | dict(_question=entry['question'])
        meta = json.loads(json.dumps(meta, default=str))
        return Problem(meta, entry['answer'])

    def score_answer(self, answer, entry):
        sd=entry['metadata']['source_dataset']
        scorer = reasoning_gym.get_score_answer_fn(sd)
        try:
            score = scorer(answer,entry)
        except Exception as e:
            print(f"Error scoring, T={entry['metadata']['task_name']} answer: {e}")
            score = 0
        return score

    def prompt(self, metadata):
        return metadata._question
        
       