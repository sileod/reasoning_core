#!/usr/bin/env python3
"""Generation worker - runs tasks directly (no internal multiprocessing)."""
import random, argparse, os, time, json, math
from pathlib import Path
from datetime import datetime

# Thread controls (must be before numpy/scipy imports)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from reasoning_core import list_tasks, get_task
import numpy as np

MEM_LIMIT_GB, MEM_WARN_RATIO = 50, 0.8
def get_rss_gb():
    try:
        with open('/proc/self/status') as f:
            for l in f:
                if l.startswith('VmRSS:'): return int(l.split()[1]) / 1024 / 1024
    except: pass
    return 0
def check_mem(log_file, worker_id, task_str):
    rss = get_rss_gb()
    if rss > MEM_LIMIT_GB * MEM_WARN_RATIO:
        with open(log_file, 'a') as f: f.write(f"Worker {worker_id} | {task_str}: MEM_WARN {rss:.1f}GB\n")

def serialize_example(example):
    row = example.to_dict()
    metadata = row.get('metadata')
    if metadata:
        row['task'] = metadata.get('source_task', row.get('task'))
        row['metadata'] = json.dumps(metadata)
    return row

def log_batch(out_path, name, level, dt, n, status):
    '''One line per BATCH attempt, including the ones that produce no rows.

    Row-level `_time` only covers accepted rows, so it silently omits rejected candidates and
    entirely-failed batches -- i.e. exactly the tasks that are expensive. This is the honest cost.
    '''
    try:
        with open(Path(out_path).parent / 'batches.jsonl', 'a') as f:
            f.write(json.dumps({'task': name, 'level': level, 'batch_time_s': round(dt, 4),
                                'rows': n, 'status': status, 'ts': time.time()}) + '\n')
    except Exception:
        pass


def run_task(name, idx, level, out_path, batch_size, max_tokens):
    '''Run a single task batch, return (success, message).'''
    t0 = time.perf_counter()
    try:
        T = get_task(name)
        T.timeout = 20 * (1 + level) ** 2
        random.seed(None)
        np.random.seed(None)

        examples = T.generate_balanced_batch(batch_size=batch_size, max_tokens=max_tokens, level=level)
        dt = time.perf_counter() - t0

        if not examples:
            log_batch(out_path, name, level, dt, 0, 'EMPTY')
            return False, 'EMPTY'

        for x in examples:
            m = x.metadata if isinstance(x.metadata, dict) else None
            if m is not None:
                m['_batch_time_s'] = round(dt, 4)
                m['_batch_n'] = len(examples)
                m['_batch_time_per_row_s'] = round(dt / len(examples), 5)

        # Serialise EVERYTHING before creating the file. The old code opened the file and then
        # serialised row by row, so a raise mid-write (coreference: 'Object of type Entity is not
        # JSON serializable') left a 0-byte .jsonl indistinguishable from a real empty batch --
        # and because the file existed, the worker never retried that batch.
        payload = ''.join(json.dumps(serialize_example(x)) + '\n' for x in examples)
        (Path(out_path) / f'{name}-{idx}.jsonl').write_text(payload)
        log_batch(out_path, name, level, dt, len(examples), 'OK')
        return True, 'OK'
    except Exception as e:
        log_batch(out_path, name, level, time.perf_counter() - t0, 0, 'ERR:' + type(e).__name__)
        return False, f'ERR: {type(e).__name__}: {e}'

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
                # Clean stale locks (older than 900s = before worker timeout)
                if lock_f.exists():
                    try:
                        if time.time() - lock_f.stat().st_mtime > 900: lock_f.unlink()
                        else: continue
                    except: continue
                
                try:
                    fd = os.open(lock_f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                except OSError:
                    continue
                
                claimed_any = True
                max_l = max(args.levels)

                custom_max = {
                    'proof_reconstruction': 2,
                    'bayesian_association': 0,
                    'bayesian_intervention': 0,
                    'logic_nli': 3,
                    'evidence_retrieval': 3,
                    'table_conversion': 4,
                }
                if d_name in custom_max:
                     max_l = min(max_l, custom_max[d_name])
  
                
                level = random.choice([l for l in args.levels if l <= max_l])
                task_str = f"{d_name}-{level}"
                t0 = time.time()
                status_file.write_text(f"Worker {args.id:>3} | {task_str:<40} | running | Done: {tasks_done:<5} | ts:{int(t0)}")
                
                try:
                    success, msg = run_task(d_name, idx, level, out_path, args.batch_size, args.max_tokens)
                    check_mem(error_log, args.id, task_str)
                    if success:
                        tasks_done += 1
                    else:
                        with open(error_log, 'a') as f: 
                            f.write(f"Worker {args.id} | {task_str}: {msg}\n")
                except Exception as e:
                    with open(error_log, 'a') as f: 
                        f.write(f"Worker {args.id} | {task_str}: CRASH: {type(e).__name__}: {e}\n")
                finally:
                    if lock_f.exists(): lock_f.unlink()
            
            if not claimed_any: break
    finally:
        if status_file.exists(): status_file.unlink()

if __name__ == '__main__':
    date = int(datetime.now().timestamp())
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_examples', default=10_000_000, type=int)
    parser.add_argument('-f', default=None)
    parser.add_argument('--id', required=True, type=str)
    parser.add_argument('--version', default=f'rc-{date}', type=str)
    parser.add_argument('--out_path', default='generated_data', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--levels', nargs='+', type=int, default=[0,1,2])
    parser.add_argument('--status_dir', required=True, type=str)
    parser.add_argument('--tasks', nargs='+', type=str, default=[])
    parser.add_argument('--max_tokens', default=5_000, type=int)
    
    args, _ = parser.parse_known_args()
    main(args)
