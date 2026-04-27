import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.api.task import ConfigurableTask
import numpy as np
from transformers import DataCollatorForSeq2Seq
from datasets import disable_progress_bar, get_dataset_config_names, load_dataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch
from tabulate import tabulate
from lm_eval.evaluator import evaluate
from lm_eval.tasks import TaskManager, get_task_dict

platinum = ['gsm8k','svamp','winograd_wsc']

platinum = [
    "drop",
    "gsm8k",
    "hotpotqa",
    "mmlu_math",
    "multiarith",
    "singleop",
    "singleq",
    "squad",
    "svamp",
    "tab_fact",
    #"vqa",
    "winograd_wsc",
    "bbh_logical_deduction_three_objects",
    "bbh_navigate",
    "bbh_object_counting",
]

harness_tasks = [
    "cola", "sst2", "mnli", "qnli", "rte", "boolq", "copa", "cb",
    "swag", "piqa", "openbookqa", "sciq", "triviaqa","arc_easy", "lambada_openai",
    "tinyMMLU", "tinyHellaswag", "tinyWinogrande", "tinyArc", "tinyGSM8k"
    ]     #social_iqa wsc tinytruthfullqa: not working 



tasksource = ['ConTRoL-nli', 'folio','anli/a1','WANLI','sick/label','glue/rte','glue/cola','cladder']

downstream_tasks = tasksource + platinum 

def load_downstream(config):
    if config in platinum:
        df = load_dataset("madrylab/platinum-bench", config, split='test')
        df = df.to_pandas()
        df=df[df.cleaning_status!='rejected']
        df['answer']=df.platinum_target
        df['prompt'] = df.platinum_prompt_no_cot
        def evaluate_row(x):
            return x.extracted in [str(x).lower() for x in x.platinum_target]

    if config in tasksource:
        ds = load_dataset("tasksource/tasksource-instruct-v0",split='validation')
        df=ds.rename_column('inputs','prompt').to_pandas()
        df = df[df.task==config]
        df.targets=df.targets.map(lambda x:x.rstrip('.'))
        if len(df)>200:
            df=df.sample(200, random_state=0)
        def evaluate_row(x):
            prepr = lambda x: str(x).lower().strip()
            return prepr(x.extracted) == prepr(x.targets)
        
    return evaluate_row, df



def run_platinum(model, tokenizer, tasks=platinum, limit=200, batch_size=16, use_chat_template=False):
    disable_progress_bar(), model.eval()
    tasks = get_dataset_config_names("madrylab/platinum-bench")
    tasks.remove('vqa')
    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
    metrics = {}

    for t in tqdm(tasks):
        ds = load_dataset("madrylab/platinum-bench", t, split=f"test[:{limit}]")
        ds = ds.filter(lambda x: x['platinum_target'] is not None)
        def process(x):
            q_text = x['platinum_prompt_no_cot'] + "\n"
            if tokenizer.chat_template and use_chat_template:
                q_ids = tokenizer.apply_chat_template([{"role":"user", "content":q_text}], tokenize=True, add_generation_prompt=True)
            else:
                q_ids = tokenizer(q_text).input_ids
            a_ids = tokenizer(x['platinum_target'][0] + tokenizer.eos_token, add_special_tokens=False).input_ids
            return {"input_ids": q_ids + a_ids, "labels": [-100]*len(q_ids) + a_ids}

        dl = DataLoader(ds.map(process, remove_columns=ds.column_names), batch_size=batch_size, collate_fn=collator)
    
        with torch.no_grad():
            losses = [model(**{k: v.to(model.device) for k,v in b.items()}).loss.item() for b in dl]
        
        metrics[f"platinum/{t}/nll"] = float(np.mean(losses))
    
    metrics['platinum/platinum_avg/nll'] = np.mean(list(metrics.values()))
    print(tabulate(metrics.items()))
    return metrics




def run_harness(model, tokenizer, limit=200):
    custom_tasks = {
        name: ConfigurableTask(config={
            "task": name, "dataset_path": path,
            "output_type": "multiple_choice",
            "test_split": "train", "doc_to_text": "",
            "doc_to_choice": '["{{sentence_good}}", "{{sentence_bad}}"]',
            "doc_to_target": 0,
            "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
        })
        for name, path in [
            ("blimp", "tasksource/blimp"),
            ("zorro", "tasksource/zorro"),
        ]
    }

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")
    
    task_manager = TaskManager()
    standard_tasks = get_task_dict(harness_tasks, task_manager)
    
    task_dict = {**standard_tasks, **custom_tasks}
    
    res = evaluate(lm=hflm, task_dict=task_dict, limit=limit)['results']
    
    s = {t: next((m[k] for k in ['mcc,none', 'acc_norm,none', 'acc,none'] if k in m), 0.) for t, m in res.items()}
    return s
    
