# generation_worker.py
from reasoning_core import DATASETS
import random
import pandas as pd
import json
from pathlib import Path
import argparse
import os
import sys
import time

# --- Argument Parsing (unchanged) ---
parser = argparse.ArgumentParser()
parser.add_argument('--num_examples', default=100_000, type=int)
parser.add_argument('-f', default=None)
parser.add_argument('--id', required=True, type=str)
parser.add_argument('--version', default='rc0',type=str)
parser.add_argument('--out_path', default='generated_data', type=str)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument("--levels", nargs="+", type=int, default=[6])
parser.add_argument('--status_dir', required=True, type=str)
args, unknown = parser.parse_known_args()

# --- Main Generation and Monitoring Logic ---
def generate_and_monitor():
    status_file = Path(args.status_dir) / f"worker_{int(args.id):03d}.status"
    tasks_completed = 0

    try:
        while True:
            out_path = Path(args.out_path) / args.version
            os.makedirs(out_path, exist_ok=True)
            blocklist = []
            tasks = [t for t in DATASETS.keys() if t.lower() not in blocklist]

            files_per_task = args.num_examples // (args.batch_size * len(tasks)) if tasks else 0
            if files_per_task < 1:
                break

            random.shuffle(tasks)
            task_to_run = None
            for name in tasks:
                index = len(list(out_path.glob(f'{name}-*.jsonl')))
                if index < files_per_task:
                    task_to_run = (name, index)
                    break
            
            if not task_to_run:
                break

            dataset_name, index = task_to_run
            
            level = random.choice(args.levels)
            if level>=4 and dataset_name in ['proof_reconstruction', 'bayesian_association', 'bayesian_intervention']:
                continue

            pid = os.fork()

            if pid > 0: # PARENT PROCESS (The Monitor)
                start_time = time.time()
                while True:
                    child_pid, exit_status = os.waitpid(pid, os.WNOHANG)
                    if child_pid == pid:
                        if exit_status == 0:
                            tasks_completed += 1
                        break

                    elapsed = time.time() - start_time
                    
                    # We combine dataset_name and level, and adjust padding to keep alignment.
                    task_label = f"{dataset_name}-{level}"
                    status_line = f"Worker {args.id:>3} | Task: {task_label:<25} | Elapsed: {elapsed:5.1f}s | Done: {tasks_completed}"
                    
                    status_file.write_text(status_line)
                    time.sleep(1)
            
            else: # CHILD PROCESS (The Worker)
                T = DATASETS[dataset_name]()
                T.timeout = 20
                
                # No longer need: level = random.choice(args.levels)
                examples = T.generate_balanced_batch(batch_size=args.batch_size, level=level)
                
                if examples:
                    d_out_path = out_path / f'{dataset_name}-{index}.jsonl'
                    df = pd.DataFrame([x.to_dict() for x in examples])
                    df['metadata'] = df['metadata'].map(json.dumps)
                    df.to_json(d_out_path, lines=True, orient='records')
                    sys.exit(0)
                else:
                    sys.exit(1)

    finally:
        if status_file.exists():
            status_file.unlink()

if __name__ == '__main__':
    generate_and_monitor()
