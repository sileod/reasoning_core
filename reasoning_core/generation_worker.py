import random, argparse, os, time, json, sys, math
import multiprocessing as mp
from pathlib import Path
from reasoning_core import list_tasks, get_task
import string
import random
import numpy as np

alphabet = string.ascii_lowercase + string.digits


def worker_loop(in_q, out_q, out_path, batch_size):
    """Worker stays alive. No re-importing overhead."""
    while True:
        task = in_q.get()
        if task is None: break 
        
        name, idx, lvl = task
        try:
            T = get_task(name)()
            T.timeout = 20 * (1+lvl)**2
            random.seed(None)
            np.random.seed(None)

            examples = T.generate_balanced_batch(batch_size=batch_size, level=lvl)

            if examples:
                # EDIT: Removed UUID. Use deterministic filename based on 'idx'.
                dest = Path(out_path) / f'{name}-{idx}.jsonl'
                with open(dest, 'w') as f:
                    for x in examples:
                        row = x.to_dict()
                        if 'metadata' in row: row['metadata'] = json.dumps(row['metadata'])
                        f.write(json.dumps(row) + '\n')
                out_q.put("OK")
            else:
                out_q.put("FAIL")
        except Exception as e:
            out_q.put(f"ERR: {e}")
def generate_and_monitor(args):
    task_q, res_q = mp.Queue(), mp.Queue()
    out_path = Path(args.out_path) / args.version
    os.makedirs(out_path, exist_ok=True)

    p = mp.Process(target=worker_loop, args=(task_q, res_q, str(out_path), args.batch_size))
    p.start()

    status_file = Path(args.status_dir) / f"worker_{int(args.id):03d}.status"
    tasks_done = 0
    #'bayesian_association','bayesian_intervention'
    blocklist = {'proof_reconstruction','float_counterfactual'}
    tasks = [t for t in (args.tasks or list_tasks()) if t.lower() not in blocklist]
            
    try:
        # EDIT: Calculate exact target and generate a randomized job list
        target_per_task = math.ceil(args.num_examples / (args.batch_size * len(tasks) or 1))
        all_jobs = [(t, i) for t in tasks for i in range(target_per_task)]
        random.shuffle(all_jobs) # Shuffle to prevent collision between processes

        for d_name, idx in all_jobs:
            
            # Paths for File Locking mechanism
            final_f = out_path / f'{d_name}-{idx}.jsonl'
            lock_f = out_path / f'{d_name}-{idx}.lock'

            # Skip if done
            if final_f.exists(): continue

            # Attempt to claim this batch index atomically
            try:
                # O_CREAT | O_EXCL ensures we only proceed if file didn't exist
                fd = os.open(lock_f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except OSError:
                continue # Locked or finished by another worker, skip

            # --- Task Level Logic ---
            max_l = max(args.levels)
            if d_name in {'proof_reconstruction', 'bayesian_association', 'bayesian_intervention','graph_isomorphism'}: max_l = min(max_l, 2)
            if d_name in {"planning", 'logic_nli','evidence_retrieval'}: max_l = min(max_l, 4)
            level = random.choice([l for l in args.levels if l <= max_l])

            task_q.put((d_name, idx, level))

            t0 = time.time()
            task_str = f"{d_name}-{level}"
            
            while res_q.empty():
                if not p.is_alive(): raise RuntimeError("Worker died")
                elapsed = time.time() - t0
                if elapsed > 0.5:
                    status_file.write_text(f"Worker {args.id:>3} | {task_str:<40} | {elapsed:5.1f}s | Done: {tasks_done:<5}")
                time.sleep(0.005)

            res = res_q.get()
            if res == "OK": 
                tasks_done += 1
            
            # Remove lock file (we either have the .jsonl now, or we failed and retry later)
            if lock_f.exists(): lock_f.unlink()

    except Exception as e:
        print(f"Error: {e}")
        exit()
    finally:
        task_q.put(None)
        p.join()
        if status_file.exists(): status_file.unlink()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_examples', default=10_000_000, type=int)
    parser.add_argument('-f', default=None)
    parser.add_argument('--id', required=True, type=str)
    parser.add_argument('--version', default='rc0', type=str)
    parser.add_argument('--out_path', default='generated_data', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 2, 4, 6])
    parser.add_argument('--status_dir', required=True, type=str)
    parser.add_argument('--tasks', nargs='+', type=str, default=[])
    args, unknown = parser.parse_known_args()

    generate_and_monitor(args)
