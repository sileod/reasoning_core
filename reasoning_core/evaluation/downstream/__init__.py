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

harness_tasks = ['leaderboard_bbh',
    "cola", "sst2", "mnli", "qnli", "rte", "boolq", "copa", "cb",'commonsense_qa',
    "swag", "piqa", "openbookqa", "sciq", "triviaqa","arc_easy",'arc_challenge', "lambada_openai","lambada_standard",
    "tinyMMLU", "tinyHellaswag", "tinyWinogrande", "tinyArc", "tinyGSM8k", "winogrande",
    "anli_r1", "anli_r2", "anli_r3",
    ]     #social_iqa wsc prost: not working
nll_metrics = ("folio", "leaderboard_bbh")

logic_custom_task_configs = {
    "wanli": {
        "task": "wanli",
        "dataset_path": "alisawuffles/WANLI",
        "validation_split": "test",
        "output_type": "multiple_choice",
        "doc_to_text": "Premise: {{premise}}\nHypothesis: {{hypothesis}}\nLabel:",
        "doc_to_choice": '["entailment", "neutral", "contradiction"]',
        "doc_to_target": '{{["entailment", "neutral", "contradiction"].index(gold)}}',
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
    },
    "hans": {
        "task": "hans",
        "dataset_path": "hans",
        "dataset_name": "plain_text",
        "validation_split": "validation",
        "output_type": "multiple_choice",
        "doc_to_text": "Premise: {{premise}}\nHypothesis: {{hypothesis}}\nLabel:",
        "doc_to_choice": '["entailment", "non-entailment"]',
        "doc_to_target": "{{label}}",
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
    },
    "nan_nli": {
        "task": "nan_nli",
        "dataset_path": "joey234/nan-nli",
        "training_split": "train",
        "test_split": "train",
        "output_type": "multiple_choice",
        "doc_to_text": "Premise: {{premise}}\nHypothesis: {{hypothesis}}\nLabel:",
        "doc_to_choice": '["entailment", "neutral", "contradiction"]',
        "doc_to_target": '{{["entailment", "neutral", "contradiction"].index(label)}}',
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
    },
    "folio": {
        "task": "folio",
        "dataset_path": "tasksource/folio",
        "validation_split": "validation",
        "output_type": "multiple_choice",
        "doc_to_text": "Premises:\n{{premises}}\nConclusion: {{conclusion}}\nLabel:",
        "doc_to_choice": '["True", "False", "Uncertain"]',
        "doc_to_target": '{{["True", "False", "Uncertain"].index(label)}}',
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
    },
    "logiqa2_nli": {
        "task": "logiqa2_nli",
        "dataset_path": "tasksource/logiqa-2.0-nli",
        "validation_split": "validation",
        "output_type": "multiple_choice",
        "doc_to_text": "Premise: {{premise}}\nHypothesis: {{hypothesis}}\nLabel:",
        "doc_to_choice": '["entailment", "not-entailment"]',
        "doc_to_target": '{{["entailment", "not-entailment"].index(label)}}',
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
    },
    "semantic_fragments_nli": {
        "task": "semantic_fragments_nli",
        "dataset_path": "tasksource/semantic_fragments_nli",
        "validation_split": "dev",
        "output_type": "multiple_choice",
        "doc_to_text": "Premise: {{sentence1}}\nHypothesis: {{sentence2}}\nLabel:",
        "doc_to_choice": '["entailment", "neutral", "contradiction"]',
        "doc_to_target": '{{["entailment", "neutral", "contradiction"].index(gold_label)}}',
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
    },
    "control_nli": {
        "task": "control_nli",
        "dataset_path": "tasksource/ConTRoL-nli",
        "validation_split": "validation",
        "output_type": "multiple_choice",
        "doc_to_text": "Premise: {{premise}}\nHypothesis: {{hypothesis}}\nLabel:",
        "doc_to_choice": '["entailment", "neutral", "contradiction"]',
        "doc_to_target": '{{["entailment", "neutral", "contradiction"].index(label)}}',
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
    },
    "boardgameqa": {
        "task": "boardgameqa",
        "dataset_path": "tasksource/Boardgame-QA",
        "validation_split": "valid",
        "output_type": "multiple_choice",
        "doc_to_text": "{{example}}\nAnswer:",
        "doc_to_choice": '["proved", "disproved", "unknown"]',
        "doc_to_target": '{{["proved", "disproved", "unknown"].index(label)}}',
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
    },
    "commonsense_qa_2": {
        "task": "commonsense_qa_2",
        "dataset_path": "tasksource/commonsense_qa_2.0",
        "validation_split": "validation",
        "output_type": "multiple_choice",
        "doc_to_text": "{{question}}\nAnswer:",
        "doc_to_choice": '["yes", "no"]',
        "doc_to_target": '{{["yes", "no"].index(answer)}}',
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
    },
    "math_qa": {
        "task": "math_qa",
        "dataset_path": "regisss/math_qa",
        "validation_split": "validation",
        "output_type": "multiple_choice",
        "doc_to_text": "{{Problem}}\n{{options}}\nAnswer:",
        "doc_to_choice": '["a", "b", "c", "d", "e"]',
        "doc_to_target": '{{["a", "b", "c", "d", "e"].index(correct)}}',
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
    },
    "gsm8k_mc": {
        "task": "gsm8k_mc",
        "dataset_path": "guipenedo/gsm8k-mc",
        "test_split": "test",
        "output_type": "multiple_choice",
        "doc_to_text": "{{Question}}\nA) {{A}}\nB) {{B}}\nC) {{C}}\nD) {{D}}\nAnswer:",
        "doc_to_choice": '["A", "B", "C", "D"]',
        "doc_to_target": '{{["A", "B", "C", "D"].index(Answer)}}',
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
    },
    "infotabs": {
        "task": "infotabs",
        "dataset_path": "table-benchmark/infotabs",
        "validation_split": "dev",
        "output_type": "multiple_choice",
        "doc_to_text": "Table title: {{table_title}}\nTable: {{table}}\nQuestion: {{question}}\nLabel:",
        "doc_to_choice": '["Entailment", "Neutral", "Contradiction"]',
        "doc_to_target": '{{["Entailment", "Neutral", "Contradiction"].index(answer)}}',
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
    },
}

custom_tasks = {
    name: {
        "task": name, "dataset_path": path,
        "output_type": "multiple_choice",
        "test_split": "train", "doc_to_text": "",
        "doc_to_choice": '["{{sentence_good}}", "{{sentence_bad}}"]',
        "doc_to_target": 0,
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
    }
    for name, path in [
        ("blimp", "tasksource/blimp"),
        ("zorro", "tasksource/zorro"),
    ]
}
default_logic_custom_task_configs = {
    name: config for name, config in logic_custom_task_configs.items()
    if name != "hans"
}
custom_tasks.update(default_logic_custom_task_configs)


def _custom_task(name):
    return ConfigurableTask(config=custom_tasks[name])

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


def _flat_tasks(task_dict):
    for name, task in task_dict.items():
        if isinstance(task, dict):
            yield from _flat_tasks(task)
        else:
            yield name, task


def _eval_docs(task):
    if task.has_validation_docs():
        return task.validation_docs()
    if task.has_test_docs():
        return task.test_docs()
    return task.training_docs()


def _target_text(task, doc):
    target = task.doc_to_target(doc)
    if isinstance(target, list):
        target = target[0]
    if isinstance(target, int) and getattr(task.config, "doc_to_choice", None) is not None:
        return task.doc_to_choice(doc)[target]
    return str(target)


def _nll_batch(ds, task, tokenizer, model, batch_size):
    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
    delimiter = getattr(task.config, "target_delimiter", " ")

    def process(doc):
        prompt = task.doc_to_text(doc)
        target = delimiter + _target_text(task, doc) + (tokenizer.eos_token or "")
        q_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        a_ids = tokenizer(target, add_special_tokens=False).input_ids
        return {"input_ids": q_ids + a_ids, "labels": [-100] * len(q_ids) + a_ids}

    dl = DataLoader([process(d) for d in ds], batch_size=batch_size, collate_fn=collator)
    with torch.no_grad():
        return [model(**{k: v.to(model.device) for k, v in b.items()}).loss.item() for b in dl]


def run_nll_metrics(model, tokenizer, tasks=nll_metrics, limit=200, batch_size=16, task_manager=None):
    disable_progress_bar(), model.eval()
    manager = task_manager
    metrics = {}
    for task_name in tqdm(tasks):
        try:
            if task_name in custom_tasks:
                task_dict = {task_name: _custom_task(task_name)}
            else:
                manager = manager or TaskManager()
                task_dict = get_task_dict([task_name], manager)
            task_losses = []
            for name, task in _flat_tasks(task_dict):
                docs = list(_eval_docs(task))[:limit]
                if not docs:
                    continue
                losses = _nll_batch(docs, task, tokenizer, model, batch_size)
                task_losses.extend(losses)
                if name == task_name:
                    metrics[f"nll/{name}/nll"] = float(np.mean(losses))
            if task_losses and f"nll/{task_name}/nll" not in metrics:
                metrics[f"nll/{task_name}/nll"] = float(np.mean(task_losses))
        except Exception as e:
            print(f"Skipping nll/{task_name}: {e}")
    return metrics






def pick_metric(m):
    return next((m[k] for k in ['mcc,none', 'acc_norm,none', 'acc,none'] if k in m), 0.)


def add_bbh0(s, hflm, task_manager, limit=200):
    def set_fewshot(task_dict, n):
        for x in task_dict.values():
            set_fewshot(x, n) if isinstance(x, dict) else x.set_config(key="num_fewshot", value=n)

    bbh = get_task_dict(["leaderboard_bbh"], task_manager)
    set_fewshot(bbh, 0)
    r = evaluate(lm=hflm, task_dict=bbh, limit=limit)['results']
    s["leaderboard_bbh_0shot"] = pick_metric(r["leaderboard_bbh"])
    return s


def run_harness(model, tokenizer, limit=200):
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")
    task_manager = TaskManager()
    s = {}

    for t in harness_tasks:
        try:
            r = evaluate(lm=hflm, task_dict=get_task_dict([t], task_manager), limit=limit)['results']
            s[f"{t}_3shot" if t == "leaderboard_bbh" else t] = pick_metric(r[t])
        except Exception as e:
            print(f"Skipping {t}: {e}")

    for t in custom_tasks:
        try:
            r = evaluate(lm=hflm, task_dict={t: _custom_task(t)}, limit=limit)['results']
            s[t] = pick_metric(r[t])
        except Exception as e:
            print(f"Skipping {t}: {e}")

    try:
        s = add_bbh0(s, hflm, task_manager, limit)
    except Exception as e:
        print(f"Skipping leaderboard_bbh_0shot: {e}")
    return {**s, **run_nll_metrics(model, tokenizer, limit=limit, task_manager=task_manager)}
