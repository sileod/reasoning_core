#!/usr/bin/env python3
"""Generation worker - runs tasks directly (no internal multiprocessing)."""
import random, argparse, os, time, json, math
from pathlib import Path

# Thread controls (must be before numpy/scipy imports)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from reasoning_core import list_tasks, get_task
import numpy as np

def run_task(name, idx, level, out_path, batch_size, max_tokens):
    """Run a single task batch, return (success, message)."""
    try:
        T = get_task(name)
        T.timeout = 20 * (1 + level) ** 2
        random.seed(None)
        np.random.seed(None)
        
        examples = T.generate_balanced_batch(batch_size=batch_size, max_tokens=max_tokens, level=level)
        
        if examples:
            dest = Path(out_path) / f'{name}-{idx}.jsonl'
            with open(dest, 'w') as f:
                for x in examples:
                    row = x.to_dict()
                    if 'metadata' in row: row['metadata'] = json.dumps(row['metadata'])
                    f.write(json.dumps(row) + '\n')
            return True, "OK"
        return False, "EMPTY"
    except BaseException as e:
        # Catch BaseException to handle TimeoutException (inherits from BaseException)
        return False, f"ERR: {type(e).__name__}: {e}"

def main(args):
    out_path = Path(args.out_path) / args.version
    out_path.mkdir(parents=True, exist_ok=True)
    error_log = Path('errors.log')
    status_file = Path(args.status_dir) / f"worker_{int(args.id):03d}.status"
    
    blocklist = {'float_counterfactual', 'theorem_premise_selection'}
    tasks = [t for t in (args.tasks or list_tasks()) if t.lower() not in blocklist]
    
    target_per_task = math.ceil(args.num_examples / (args.batch_size * len(tasks) or 1))
    all_jobs = [(t, i) for t in tasks for i in range(target_per_task)]
    random.shuffle(all_jobs)
    
    tasks_done = 0
    try:
        while True:
            claimed_any = False
            for d_name, idx in all_jobs:
                final_f = out_path / f'{d_name}-{idx}.jsonl'
                lock_f = out_path / f'{d_name}-{idx}.lock'
                
                if final_f.exists(): continue
                
                try:
                    fd = os.open(lock_f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                except OSError:
                    continue
                
                claimed_any = True
                max_l = max(args.levels)
                if d_name in {'proof_reconstruction', 'bayesian_association', 'bayesian_intervention', 'graph_isomorphism'}: 
                    max_l = min(max_l, 2)
                if d_name in {'planning', 'logic_nli', 'evidence_retrieval'}: 
                    max_l = min(max_l, 4)
                
                level = random.choice([l for l in args.levels if l <= max_l])
                task_str = f"{d_name}-{level}"
                t0 = time.time()
                status_file.write_text(f"Worker {args.id:>3} | {task_str:<40} | running | Done: {tasks_done:<5} | ts:{int(t0)}")
                
                try:
                    success, msg = run_task(d_name, idx, level, out_path, args.batch_size, args.max_tokens)
                    
                    if success:
                        tasks_done += 1
                    else:
                        with open(error_log, 'a') as f: 
                            f.write(f"Worker {args.id} | {task_str}: {msg}\n")
                except BaseException as e:
                    # Catch ALL exceptions including TimeoutException (inherits BaseException)
                    # to prevent worker death - just log and continue to next job
                    with open(error_log, 'a') as f: 
                        f.write(f"Worker {args.id} | {task_str}: CRASH: {type(e).__name__}: {e}\n")
                finally:
                    if lock_f.exists(): lock_f.unlink()
            
            if not claimed_any: break
    finally:
        if status_file.exists(): status_file.unlink()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_examples', default=10_000_000, type=int)
    parser.add_argument('-f', default=None)
    parser.add_argument('--id', required=True, type=str)
    parser.add_argument('--version', default='rc0', type=str)
    parser.add_argument('--out_path', default='generated_data', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--levels', nargs='+', type=int, default=[0, 2, 4])
    parser.add_argument('--status_dir', required=True, type=str)
    parser.add_argument('--tasks', nargs='+', type=str, default=[])
    parser.add_argument('--max_tokens', default=5_000, type=int)
    
    args, _ = parser.parse_known_args()
    main(args)
