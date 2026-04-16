from reasoning_core import list_tasks, get_task
import time

failed =[]
for t in list_tasks():
    t0=time.time()
    print(t.ljust(30, '.'), end="")
    try:
        get_task(t).validate()
        print(f"{time.time() - t0:.5f}")
    except Exception as e:
        failed+=[t]
        print(e)

print(f'Done, failed: {failed}')