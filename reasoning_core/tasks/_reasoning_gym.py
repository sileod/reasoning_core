from reasoning_core.template import Task, Problem
import reasoning_gym
import random
import json

class RG(Task):
    def __init__(self, config=dict):
        super().__init__(config)
        
    def generate(self):
        meta = dict()
        tasks = list(reasoning_gym.factory.DATASETS.keys())
        task = random.choice(tasks)
        entry=reasoning_gym.create_dataset(task, size=1, seed=None)[0]
        meta = entry['metadata'] | dict(task_name=f"RG.{task}") | dict(_question=entry['question'])
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
        
#del DATASETS['rg'] # unregister this dataset
       