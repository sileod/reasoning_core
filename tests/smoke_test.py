from reasoning_core import list_tasks, get_task
import time

failed = []
for t in list_tasks():
    t0 = time.time()
    try:
        get_task(t).validate()
        print(f"{t.ljust(30, '.')} {time.time() - t0:.5f}")
    except Exception as e:
        print(f"{t.ljust(30, '.')} EXCEPTION: {e}")
        failed.append(t)

if failed:
    raise RuntimeError(f"Failed tasks: {failed}")